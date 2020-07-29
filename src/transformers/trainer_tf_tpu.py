# Copied from https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_tf.py
# But with changes for making it work on TPU. These changes have open PR in the transformers repo, but are not yet merged

"""Tensorflow trainer class."""

import logging
import math
import os
from typing import Callable, Dict, Optional, Tuple
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from transformers.modeling_tf_utils import TFPreTrainedModel
from transformers.optimization_tf import GradientAccumulator, AdamWeightDecay
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, is_wandb_available
from transformers.training_args_tf import TFTrainingArguments

if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)


class TFTPUTrainer:
    model: TFPreTrainedModel
    args: TFTrainingArguments
    train_dataset: Optional[tf.data.Dataset]
    eval_dataset: Optional[tf.data.Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional[tf.summary.SummaryWriter] = None
    optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None
    global_step: Optional[int] = None
    epoch_logging: Optional[float] = None

    def __init__(
            self,
            model: TFPreTrainedModel,
            args: TFTrainingArguments,
            train_dataset: Optional[tf.data.Dataset] = None,
            eval_dataset: Optional[tf.data.Dataset] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            prediction_loss_only=False,
            tb_writer: Optional[tf.summary.SummaryWriter] = None,
            optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        self.gradient_accumulator = GradientAccumulator()
        self.global_step = 0
        self.epoch_logging = 0

        if tb_writer is not None:
            self.tb_writer = tb_writer
        else:
            self.tb_writer = tf.summary.create_file_writer(self.args.logging_dir)
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )

    def _compute_steps(self, ds, epochs=1):
        num_train_examples = ds.reduce(tf.constant(0), lambda x, _: x + 1).numpy()

        if self.args.dataloader_drop_last:
            approx_fn = math.floor
        else:
            approx_fn = math.ceil
        train_steps = approx_fn(
            num_train_examples / (self.args.train_batch_size * self.args.gradient_accumulation_steps))

        if self.args.max_steps > 0:
            total_train_steps = self.args.max_steps
        else:
            total_train_steps = train_steps * self.args.num_train_epochs

        return num_train_examples, train_steps, total_train_steps

    def get_train_tfdataset(self) -> tf.data.Dataset:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        self.num_train_examples, self.train_steps, self.total_train_steps = self._compute_steps(self.train_dataset)

        ds = (
            self.train_dataset.cache()
                .shuffle(self.num_train_examples)
                .batch(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                       drop_remainder=self.args.dataloader_drop_last)
                .unbatch()
                .batch(self.args.train_batch_size, drop_remainder=self.args.dataloader_drop_last)
                .prefetch(tf.data.experimental.AUTOTUNE)
                # .repeat()
        )

        return self.args.strategy.experimental_distribute_dataset(ds)

    def get_eval_tfdataset(self, eval_dataset: Optional[tf.data.Dataset] = None) -> tf.data.Dataset:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        _, steps, _ = self._compute_steps(eval_dataset)

        ds = (
            eval_dataset.cache()
                .batch(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                       drop_remainder=self.args.dataloader_drop_last)
                .unbatch()
                .batch(self.args.train_batch_size, drop_remainder=self.args.dataloader_drop_last)
                .prefetch(tf.data.experimental.AUTOTUNE)
                .repeat()
        )

        return self.args.strategy.experimental_distribute_dataset(ds), steps

    def set_optimizers(
            self,
    ) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is None:
            self.optimizers = create_optimizer(
                self.args.learning_rate,
                self.total_train_steps,
                self.args.warmup_steps,
                adam_epsilon=self.args.adam_epsilon,
                weight_decay_rate=self.args.weight_decay,
            )

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.
        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:
        Environment:
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))

    @tf.function
    def _evaluate_steps(self, per_replica_features, per_replica_labels):
        """
        One step evaluation across replica.
        Args:
          per_replica_features: the batched features.
          per_replica_labels: the batched labels.
        Returns:
          The loss corresponding to the given batch.
        """
        per_replica_loss, per_replica_logits = self.args.strategy.experimental_run_v2(
            self._run_model, args=(per_replica_features, per_replica_labels, False)
        )

        if isinstance(per_replica_loss, tuple):
            reduced_loss = tuple(self._reduce_loss(l) for l in per_replica_loss)
        else:
            reduced_loss = self._reduce_loss(per_replica_loss)

        return reduced_loss, per_replica_logits

    def _prediction_loop(
            self, raw_dataset: tf.data.Dataset, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """
        # Create the dataset and retrieve the number of steps to iterate
        dataset, eval_steps = self.get_eval_tfdataset(raw_dataset)

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        logger.info("***** Running %s *****", description)
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        label_ids: np.ndarray = None
        preds: np.ndarray = None

        step: int = 1

        dataset_iterator = iter(dataset)

        while True:
            features, labels = next(dataset_iterator)

            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss, logits = self._evaluate_steps(features, labels)
            loss = self._loss_to_scalar(loss)

            if not prediction_loss_only:
                if isinstance(logits, tuple):
                    logits = logits[0]

                if isinstance(labels, tuple):
                    labels = labels[0]

                if self.args.n_gpu > 1:
                    for val in logits.values:
                        if preds is None:
                            preds = val.numpy()
                        else:
                            preds = np.append(preds, val.numpy(), axis=0)

                    for val in labels.values:
                        if label_ids is None:
                            label_ids = val.numpy()
                        else:
                            label_ids = np.append(label_ids, val.numpy(), axis=0)
                else:
                    if preds is None:
                        preds = logits.numpy()
                    else:
                        preds = np.append(preds, logits.numpy(), axis=0)

                    if label_ids is None:
                        label_ids = labels.numpy()
                    else:
                        label_ids = np.append(label_ids, labels.numpy(), axis=0)

            step += 1

            if step % eval_steps == 0:
                break

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        metrics["eval_loss"] = loss

        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def _loss_to_scalar(self, loss):
        if isinstance(loss, tuple):
            losses = tuple(tf.reduce_mean(l).numpy() for l in loss)
            return sum(losses)
        else:
            return tf.reduce_mean(loss).numpy()

    def _log(self, logs: Dict[str, float]) -> None:
        if self.tb_writer:
            with self.tb_writer.as_default():
                for k, v in logs.items():
                    tf.summary.scalar(k, v, step=self.global_step)
            self.tb_writer.flush()
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        logger.info(output)

    def evaluate(
            self, eval_dataset: Optional[tf.data.Dataset] = None, prediction_loss_only: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        """
        output = self._prediction_loop(eval_dataset, description="Evaluation")

        logs = {**output.metrics}
        logs["epoch"] = self.epoch_logging
        self._log(logs)

        return output.metrics

    def train(self) -> None:
        """
        Train method to train the model.
        """
        train_ds = self.get_train_tfdataset()

        if self.args.debug:
            tf.summary.trace_on(graph=True, profiler=True)

        self.gradient_accumulator.reset()

        with self.args.strategy.scope():
            self.set_optimizers()
            iterations = self.optimizers[0].iterations

        if iterations.numpy() > 0:
            logger.info("Start the training from the last checkpoint")

        tf.summary.experimental.set_step(iterations)

        if self.args.fp16:
            policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
            tf.keras.mixed_precision.experimental.set_policy(policy)

        with self.tb_writer.as_default():
            tf.summary.text("args", self.args.to_json_string())

        self.tb_writer.flush()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_train_examples)
        logger.info("  Num Epochs = %d", self.total_train_steps / self.train_steps)
        logger.info("  Total optimization steps = %d", self.total_train_steps)
        for epoch_iter in range(1, int(self.args.num_train_epochs + 1)):
            for step, training_loss in tqdm(enumerate(self._training_steps(train_ds)), position=0, total=self.total_train_steps/self.args.num_train_epochs):
                self.global_step = iterations.numpy()
                self.epoch_logging = (step + 1) / self.train_steps

                if self.args.debug:
                    logs = {}
                    logs.update(self.log_loss(training_loss))
                    logs["epoch"] = self.epoch_logging
                    self._log(logs)

                if self.global_step == 1 and self.args.debug:
                    with self.tb_writer.as_default():
                        tf.summary.trace_export(
                            name="training", step=self.global_step, profiler_outdir=self.args.logging_dir
                        )

                if self.args.evaluate_during_training and self.global_step % self.train_steps == 0:
                    result = self.evaluate()
                    if self.args.save_steps < 0:
                        ckpt_path = os.path.join(self.args.output_dir,
                                                 "ckpt_e{}_l{:.4f}".format(int(self.epoch_logging), result["eval_loss"]))
                        self.save_model(ckpt_path)

                if self.global_step % self.args.logging_steps == 0:
                    logs = {}
                    logs.update(self.log_loss(training_loss))
                    logs["learning_rate"] = self.optimizers[1](self.global_step).numpy()
                    logs["epoch"] = self.epoch_logging
                    self._log(logs)

                if self.global_step % self.total_train_steps == 0:
                    break

    def log_loss(self, loss):
        logs = {}

        if isinstance(loss, tuple):
            total_loss = 0
            for i, l in enumerate(loss):
                logs["loss_{}".format(i + 1)] = l.numpy()
                total_loss += logs["loss_{}".format(i + 1)]
            logs["loss"] = total_loss
        else:
            logs["loss"] = loss.numpy()

        return logs

    def _training_steps(self, ds):
        """
        Returns a generator over training steps (i.e. parameters update).
        """
        for i, loss in enumerate(self._accumulate_next_gradients(ds)):
            if i % self.args.gradient_accumulation_steps == 0:
                self._apply_gradients()
                yield loss

    @tf.function
    def _apply_gradients(self):
        """Applies the gradients (cross-replica)."""
        self.args.strategy.experimental_run_v2(self._step)

    def _step(self):
        """Applies gradients and resets accumulation."""
        gradient_scale = self.gradient_accumulator.step * self.args.strategy.num_replicas_in_sync
        gradients = [
            gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in self.gradient_accumulator.gradients
        ]
        gradients = [(tf.clip_by_value(grad, -self.args.max_grad_norm, self.args.max_grad_norm)) for grad in gradients]

        self.optimizers[0].apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        self.gradient_accumulator.reset()

    def _accumulate_next_gradients(self, ds):
        """Accumulates the gradients from the next element in dataset."""
        iterator = iter(ds)
        while True:
            per_replica_features, per_replica_labels = next(iterator)
            yield self._accumulate_gradients(per_replica_features, per_replica_labels)

    @tf.function
    def _accumulate_gradients(self, per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss = self.args.strategy.experimental_run_v2(
            self._forward, args=(per_replica_features, per_replica_labels)
        )

        if isinstance(per_replica_loss, tuple):
            reduced_loss = tuple(self._reduce_loss(l) for l in per_replica_loss)
        else:
            reduced_loss = self._reduce_loss(per_replica_loss)

        return reduced_loss

    def _reduce_loss(self, loss):
        try:
            return self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=0)
        except ValueError:
            return self.args.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, None)

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss, _ = self._run_model(features, labels, True)
        ys = list(per_example_loss) if isinstance(per_example_loss, tuple) else per_example_loss
        gradients = tf.gradients(ys, self.model.trainable_variables)
        gradients = [
            g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
        ]

        self.gradient_accumulator(gradients)

        return per_example_loss

    def _run_model(self, features, labels, training):
        """
        Computes the loss of the given features and labels pair.
        Args:
          features: the batched features.
          labels: the batched labels.
          training: run the model in training mode or not
        """
        if isinstance(labels, (dict)):
            loss, logits = self.model(features, training=training, **labels)[:2]
        else:
            loss, logits = self.model(features, labels=labels, training=training)[:2]

        if self.model.losses:
            if isinstance(loss, tuple):
                loss += (sum(self.model.losses) * (1.0 / self.args.n_gpu),)
            else:
                loss += sum(self.model.losses) * (1.0 / self.args.n_gpu)

        return loss, logits

    def predict(self, test_dataset: tf.data.Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        Args:
          test_dataset: something similar to a PT Dataset. This is just
            temporary before to have a framework-agnostic approach for datasets.
        """
        return self._prediction_loop(test_dataset, description="Prediction")

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the pretrained model.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        logger.info("Saving model in {}".format(output_dir))

        if not isinstance(self.model, TFPreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        self.model.save_pretrained(output_dir)  # MODIFIED