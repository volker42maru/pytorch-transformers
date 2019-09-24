# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import os
import argparse
import torch
import numpy as np
import tensorflow as tf
from pytorch_transformers import BertModel, BertConfig, BertForQuestionAnswering, \
    DistilBertModel, DistilBertConfig, DistilBertForQuestionAnswering, \
    RobertaModel, RobertaConfig, RobertaForQuestionAnswering


def convert_pytorch_checkpoint_to_tf(model:BertModel, ckpt_dir:str, model_name:str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "ffn.lin"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('token_type_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('LayerNorm/weight', 'LayerNorm/gamma'),
        ('LayerNorm/bias', 'LayerNorm/beta'),
        ('weight', 'kernel'),
        ('transformer', 'encoder'),
        ('q_lin', 'self/query'),
        ('k_lin', 'self/key'),
        ('v_lin', 'self/value'),
        ('out_lin', 'output/dense'),
        ('output_layer_norm/kernel', 'output/LayerNorm/gamma'),
        ('output_layer_norm/bias', 'output/LayerNorm/beta'),
        ('sa_layer_norm/kernel', 'attention/output/LayerNorm/gamma'),
        ('sa_layer_norm/bias', 'attention/output/LayerNorm/beta'),
        ('ffn/lin1', 'intermediate/dense'),
        ('ffn/lin2', 'output/dense'),
        ('qa_outputs/kernel', 'cls/squad/output_weights'),
        ('qa_outputs/bias', 'cls/squad/output_bias'),
        ('distilbert/', ''),
        ('roberta/', ''),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name:str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return '{}'.format(name) if 'squad' in name else 'bert/{}'.format(name)

    def create_tf_var(tensor:np.ndarray, name:str, session:tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            # print(tf_var)
            print("Pytorch var name: {}".format(var_name))
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))
            output_var = tf_name

        from tensorflow.python.tools import freeze_graph
        graph_file = os.path.join(ckpt_dir, 'model.graph')
        # tmp_g = tf.get_default_graph().as_graph_def()
        tmp_g = session.graph_def
        tmp_g = tf.graph_util.convert_variables_to_constants(session, tmp_g, [output_var])
        with tf.gfile.GFile(graph_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))



def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",
                        type=str,
                        required=True,
                        help="model type e.g. bert, distil_bert")
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="model name e.g. bert-base-uncased")
    parser.add_argument("--cache_dir",
                        type=str,
                        default=None,
                        required=False,
                        help="Directory containing pytorch model")
    parser.add_argument("--pytorch_model_path",
                        type=str,
                        required=True,
                        help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir",
                        type=str,
                        required=True,
                        help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)

    type_to_model = {'bert': (BertModel, BertConfig),
                     'bert_qa': (BertForQuestionAnswering, BertConfig),
                     'distil_bert': (DistilBertModel, DistilBertConfig),
                     'distil_bert_qa': (DistilBertForQuestionAnswering, DistilBertConfig),
                     'roberta': (RobertaModel, RobertaConfig),
                     'roberta_qa': (RobertaForQuestionAnswering, RobertaConfig)}

    model, config = type_to_model[args.model_type]

    # model = model.from_pretrained(
    #     pretrained_model_name_or_path=args.model_name,
    #     state_dict=torch.load(args.pytorch_model_path),
    #     cache_dir=args.cache_dir
    # )

    # config_path = os.path.join(args.cache_dir, 'config.json')
    config = config.from_pretrained(args.cache_dir)
    model = model.from_pretrained(args.cache_dir, config=config)
    # model = model.from_pretrained(args.model_name)

    convert_pytorch_checkpoint_to_tf(
        model=model,
        ckpt_dir=args.tf_cache_dir,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
