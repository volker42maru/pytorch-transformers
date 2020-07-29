# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import logging
import math
import os
import tensorflow as tf
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    # TextDataset,
    set_seed,
    TFAutoModelWithLMHead,
    TFTrainer,
    TFTrainingArguments
)
from transformers.trainer_tf_tpu import TFTPUTrainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class LineByLineTextDatasetTF:
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with tf.io.gfile.GFile(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)

        self.features = []
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        # print(len(batch_encoding['input_ids']))
        # for inp in batch_encoding['input_ids']:
        #     print(len(inp))
        input_ids, labels = data_collator.mask_tokens(torch.tensor(batch_encoding['input_ids']))
        input_ids = input_ids.numpy()
        labels = labels.numpy()
        for i, l in zip(input_ids, labels):
            self.features.append({
                "input_ids": i,
                "labels": l
            })

        # self.features = batch_encoding

    def get_dataset_from_generator(self):
        train_types = ({"input_ids": tf.int32}, {"labels": tf.int32})
        train_shapes = ({"input_ids": tf.TensorShape([None])}, {"labels": tf.TensorShape([None])})

        def gen():
            for i, ex in enumerate(self.features):
                yield ({"input_ids": ex["input_ids"]}, {"labels": ex["labels"]})

        return tf.data.Dataset.from_generator(gen, train_types, train_shapes)


from filelock import FileLock
import pickle
import time
class TextDataset(torch.utils.data.dataset.Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        # assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        # with FileLock(lock_path):
        #
        #     if os.path.exists(cached_features_file) and not overwrite_cache:
        #         start = time.time()
        #         with tf.io.gfile.GFile(cached_features_file, "rb") as handle:
        #             self.examples = pickle.load(handle)
        #         logger.info(
        #             f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
        #         )
        #
        #     else:
        logger.info(f"Creating features from dataset file at {directory}")

        self.examples = []
        with tf.io.gfile.GFile(file_path) as f:
            text = f.read()

        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(
                tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
            )
        # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should loook for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

        start = time.time()
        with tf.io.gfile.GFile(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
        )


# def get_dataset_from_generator(examples, tokenizer):

    # train_types = ({"input_ids": tf.int32}, {"labels": tf.int32})
    # train_shapes = ({"input_ids": tf.TensorShape([None])}, {"labels": tf.TensorShape([None])})
    #
    # def gen():
    #     for i, ex in enumerate(features):
    #         yield ({"input_ids": ex["input_ids"]}, {"labels": ex["labels"]})

    # dataset = tf.data.Dataset.from_generator(gen, train_types, train_shapes)
    # print(list(dataset.as_numpy_iterator())[:10])

    # features_dataset = tf.data.Dataset.from_tensor_slices({'input_ids': tf.constant(input_ids)})
    # labels_dataset = tf.data.Dataset.from_tensor_slices({'labels': tf.constant(labels)})
    # dataset =

    # return tf.data.Dataset.from_generator(gen, train_types, train_shapes)


import collections
class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training=False):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature["input_ids"])
        features["labels"] = create_int_feature(feature["labels"])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)
    # print(example.items())

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            tf.cast(t, tf.int32)
        example[name] = t

    # example = (example.items())

    return example


def tuplize(example):
    outputs = example.pop('labels', None)
    return example, outputs


def get_tfrecord_dataset(input_file, name_to_features):
    """The actual input function."""
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    print(f'reading tfrecord file from: {input_file}')
    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: _decode_record(record, name_to_features),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.map(tuplize)

    return d


def get_dataset_from_slice(examples, tokenizer, output_file, block_size):
    tfrecord_file = output_file + '.tfrecord'

    if not tf.io.gfile.exists(tfrecord_file):
        features = []
        data_collator = DataCollatorForLanguageModeling(tokenizer)
        input_ids, labels = data_collator.mask_tokens(torch.tensor(examples))
        input_ids = input_ids.numpy()
        labels = labels.numpy()

        for i, l in zip(input_ids, labels):
            features.append({
                "input_ids": i,
                "labels": l
            })

        print(f'writing features to tfrecord')
        feature_writer = FeatureWriter(tfrecord_file)
        for f in features:
            feature_writer.process_feature(f)
        feature_writer.close()
    else:
        print('loading from existing tfrecord file')

    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([block_size], tf.int64),
        "labels": tf.io.FixedLenFeature([block_size], tf.int64),
    }

    return get_tfrecord_dataset(tfrecord_file, name_to_features)

    # return tf.data.Dataset.from_tensor_slices(({'input_ids': tf.constant(input_ids)},
    #                                            {'labels': tf.constant(labels)}))


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        raise NotImplementedError("line_by_line dataset NOT supported right now")
        # return LineByLineTextDatasetTF(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size).get_dataset_from_generator()
    else:
        examples = TextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
        ).examples
        dataset = get_dataset_from_slice(examples, tokenizer, output_file=file_path, block_size=args.block_size)
        # print(list(dataset.as_numpy_iterator())[:10])
        return dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    with training_args.strategy.scope():
        if model_args.model_name_or_path:
            model = TFAutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = TFAutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    # with training_args.strategy.scope():
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    # if config.model_type == "xlnet":
    #     data_collator = DataCollatorForPermutationLanguageModeling(
    #         tokenizer=tokenizer, plm_probability=data_args.plm_probability, max_span_length=data_args.max_span_length,
    #     )
    # else:
    #     data_collator = DataCollatorForLanguageModeling(
    #         tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    #     )

    from transformers.optimization_tf import create_optimizer

    optimizers = create_optimizer(
        5e-5,
        4691,
        100,
        adam_epsilon=1e-8,
        weight_decay_rate=0.0,
    )

    # Initialize our Trainer
    trainer = TFTPUTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        optimizers=optimizers
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    # results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     eval_output = trainer.evaluate()
    #
    #     perplexity = math.exp(eval_output["eval_loss"])
    #     result = {"perplexity": perplexity}
    #
    #     output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    #     if trainer.is_world_master():
    #         with open(output_eval_file, "w") as writer:
    #             logger.info("***** Eval results *****")
    #             for key in sorted(result.keys()):
    #                 logger.info("  %s = %s", key, str(result[key]))
    #                 writer.write("%s = %s\n" % (key, str(result[key])))
    #
    #     results.update(result)

    # return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
