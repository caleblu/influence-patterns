# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import network  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations
from official.nlp.modeling import layers
import copy


@tf.keras.utils.register_keras_serializable(package='Text')
class TransformerEncoder(tf.keras.Model):
    """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".

  Arguments:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    sequence_length: The sequence length that this encoder expects. If None, the
      sequence length is dynamic; if an integer, the encoder will require
      sequences padded to this length.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    intermediate_size: The intermediate size for the transformer layers.
    activation: The activation to use for the transformer layers.
    dropout_rate: The dropout rate to use for the transformer layers.
    attention_dropout_rate: The dropout rate to use for the attention layers
      within the transformer layers.
    initializer: The initialzer to use for all weights in this encoder.
    return_all_encoder_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers.
  """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_layers=12,
                 num_attention_heads=12,
                 sequence_length=512,
                 max_sequence_length=None,
                 type_vocab_size=16,
                 intermediate_size=3072,
                 activation=activations.gelu,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
                 return_all_encoder_outputs=True,
                 use_stop_gradient=False,
                 model_compression=True,
                 precision=tf.float32,
                 **kwargs):
        activation = tf.keras.activations.get(activation)
        initializer = tf.keras.initializers.get(initializer)
        if not max_sequence_length:
            max_sequence_length = sequence_length
        self._self_setattr_tracking = False
        self._config_dict = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'sequence_length': sequence_length,
            'max_sequence_length': max_sequence_length,
            'type_vocab_size': type_vocab_size,
            'intermediate_size': intermediate_size,
            'activation': tf.keras.activations.serialize(activation),
            'dropout_rate': dropout_rate,
            'attention_dropout_rate': attention_dropout_rate,
            'initializer': tf.keras.initializers.serialize(initializer),
            'return_all_encoder_outputs': return_all_encoder_outputs,
        }

        # word_ids = tf.keras.layers.Input(shape=(sequence_length,),
        #                                  dtype=tf.int32,
        #                                  name='input_word_ids')
        mask = tf.keras.layers.Input(shape=(sequence_length,),
                                     dtype=tf.int32,
                                     name='input_mask')
        type_ids = tf.keras.layers.Input(shape=(sequence_length,),
                                         dtype=tf.int32,
                                         name='input_type_ids')
        word_embeddings = tf.keras.layers.Input(shape=(sequence_length,
                                                       hidden_size),
                                                dtype=precision,
                                                name='word_embeddings')
        # attention_
        # word_embeddings_end = tf.keras.layers.Input(shape=(sequence_length,
        #                                                    hidden_size),
        #                                             dtype=precision,
        #                                             name='word_embeddings_end')
        # word_embeddings_baseline = tf.keras.layers.Input(
        #     shape=(sequence_length, hidden_size),
        #     dtype=precision,
        #     name='word_embeddings_baseline')

        # alpha = tf.keras.layers.Input(shape=(1,), dtype=precision, name='alpha')
        if use_stop_gradient or model_compression:
            ind = tf.keras.layers.Input(shape=(num_layers + 1, sequence_length,
                                               hidden_size),
                                        dtype=precision,
                                        name='ind')
            ind_attention = tf.keras.layers.Input(
                shape=(num_layers, sequence_length, num_attention_heads,
                       int(hidden_size / num_attention_heads)),
                dtype=precision,
                name='ind_attention')
            include_skip_gradient = tf.keras.layers.Input(
                shape=(num_layers,),
                batch_size=1,
                dtype=precision,
                name='include_skip_gradient')
        if model_compression:
            ind_rep = tf.keras.layers.Input(shape=(num_layers + 1,
                                                   sequence_length,
                                                   hidden_size),
                                            dtype=precision,
                                            name='ind_rep')
            ind_attention_rep = tf.keras.layers.Input(
                shape=(num_layers, sequence_length, num_attention_heads,
                       int(hidden_size / num_attention_heads)),
                dtype=precision,
                name='ind_attention_rep')
            include_skip_gradient_rep = tf.keras.layers.Input(
                shape=(num_layers, sequence_length, hidden_size),
                dtype=precision,
                name='include_skip_gradient_rep')

        # word_embeddings = word_embeddings_baseline + tf.multiply(
        #     tf.expand_dims(alpha, 1),
        #     word_embeddings_end - word_embeddings_baseline)

        # Always uses dynamic slicing for simplicity.
        self._position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_sequence_length=max_sequence_length)

        position_embeddings = self._position_embedding_layer(word_embeddings)

        type_embeddings = (layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=hidden_size,
            initializer=initializer,
            use_one_hot=True,
            name='type_embeddings')(type_ids))

        embeddings = tf.keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings])

        embeddings = (tf.keras.layers.LayerNormalization(
            name='embeddings/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=precision)(embeddings))
        embeddings = (tf.keras.layers.Dropout(rate=dropout_rate)(embeddings))

        self._transformer_layers = []
        data = embeddings
        encoder_outputs = [word_embeddings]
        attention_outputs = []
        if use_stop_gradient:
            part_data = data * ind[:, 0, :, :]
            data = part_data + tf.stop_gradient(-part_data + data)
        if model_compression:
            neg_ind = tf.abs(ind[:, 0, :, :] - 1)
            data = data * ind[:, 0, :, :] + ind_rep[:, 0, :, :] * neg_ind
        encoder_outputs.append(data)
        attention_mask = layers.SelfAttentionMask()([data, mask])
        attention_probs_lst = []
        for i in range(num_layers):
            layer = layers.Transformer(
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                intermediate_activation=activation,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                kernel_initializer=initializer,
                name='transformer/layer_%d' % i)
            self._transformer_layers.append(layer)
            layer.build([data.shape, attention_mask.shape])

            input_tensor, attention_output, attention_probs = layer.get_attention_output(
                [data, attention_mask])

            attention_probs_lst.append(attention_probs)
            if use_stop_gradient:

                part_attn_output = attention_output * ind_attention[:,
                                                                    i, :, :, :]
                attention_output = part_attn_output + tf.stop_gradient(
                    -part_attn_output + attention_output)
            if model_compression:
                neg_ind = tf.abs(ind_attention[:, i, :, :, :] - 1)
                attention_output = attention_output * ind_attention[:,
                                                                    i, :, :, :] + ind_attention_rep[:,
                                                                                                    i, :, :, :] * neg_ind
            attention_outputs.append(attention_output)

            if model_compression:
                neg_ind = tf.abs(include_skip_gradient_rep[:, i, :, :] - 1)
                input_tensor = input_tensor * include_skip_gradient_rep[:,
                                                                        i, :, :] + ind_rep[:,
                                                                                           i, :, :] * neg_ind
                data = layer.get_final_output(
                    input_tensor,
                    attention_output,
                    include_skip_gradient=include_skip_gradient[0, i])
            elif use_stop_gradient:
                data = layer.get_final_output(
                    input_tensor,
                    attention_output,
                    include_skip_gradient=include_skip_gradient[0, i])
            else:
                data = layer.get_final_output(input_tensor, attention_output)

            if use_stop_gradient:
                part_data = data * ind[:, i + 1, :, :]
                data = part_data + tf.stop_gradient(-part_data + data)
            if model_compression:
                neg_ind = tf.abs(ind[:, i + 1, :, :] - 1)
                data = data * ind[:, i + 1, :, :] + ind_rep[:, i +
                                                            1, :, :] * neg_ind
            encoder_outputs.append(data)
        first_token_tensor = (
            tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
                encoder_outputs[-1]))
        cls_output = tf.keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            kernel_initializer=initializer,
            name='pooler_transform')(first_token_tensor)
        # print(first_token_tensor)
        if return_all_encoder_outputs:
            outputs = [
                encoder_outputs, attention_outputs, attention_probs_lst,
                cls_output
            ]
        else:
            outputs = [encoder_outputs[-1]]
        if use_stop_gradient:
            super(TransformerEncoder, self).__init__(inputs=[
                mask, type_ids, word_embeddings, ind, ind_attention,
                include_skip_gradient
            ],
                                                     outputs=outputs,
                                                     **kwargs)
        elif model_compression:
            super(TransformerEncoder, self).__init__(inputs=[
                mask, type_ids, word_embeddings, ind, ind_attention,
                include_skip_gradient, ind_rep, ind_attention_rep,
                include_skip_gradient_rep
            ],
                                                     outputs=outputs,
                                                     **kwargs)
        else:
            super(TransformerEncoder,
                  self).__init__(inputs=[mask, type_ids, word_embeddings],
                                 outputs=outputs,
                                 **kwargs)

    def get_config(self):
        return self._config_dict

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
