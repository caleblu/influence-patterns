import argparse
import logging
import os
import pickle as pkl

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from examples import Example, examples_in_batches
from metadata_analysis import *
from official.nlp.bert import tokenization
from utils import load_marvin, write_pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from configs import USERNAME


class InfluenceExtractor():

    def __init__(self,
                 bert_config,
                 embedding_model,
                 transformer_encoder,
                 decoder_model,
                 tokenizer,
                 use_stop_gradient=False,
                 model_compression=False,
                 decoder_type='mlm',
                 precision=tf.float32):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.embedding_mat = self.embedding_model.layers[1].weights[0]
        self.mask_embedding = self.get_embedding_from_word(
            '[MASK]')  ## baseline emb of ['mask']
        self.cls_embedding = self.get_embedding_from_word(
            '[CLS]')  ## baseline emb of ['CLS']
        self.sep_embedding = self.get_embedding_from_word(
            '[SEP]')  ## baseline emb of ['SEP']
        self.pad_embedding = self.get_embedding_from_word(
            '[PAD]')  ## baseline emb of ['PAD']
        self.transformer_encoder = transformer_encoder
        self.decoder_model = decoder_model
        self.bert_config = bert_config
        self.hidden_size = self.bert_config.hidden_size
        self.num_hidden_layers = self.bert_config.num_hidden_layers
        self.num_attention_heads = self.bert_config.num_attention_heads
        self.attention_size = int(self.hidden_size / self.num_attention_heads)
        self.use_stop_gradient = use_stop_gradient
        self.model_compression = model_compression
        self.seq_length = self.transformer_encoder.input[0].shape[1]
        self.indice_array = np.eye(self.seq_length)[np.newaxis,
                                                    np.newaxis, :, :]
        self.indice_attn_array = self.indice_array[..., np.newaxis].repeat(
            self.num_attention_heads, -1)
        self.decoder_type = decoder_type
        self.precision = precision
        assert not (self.use_stop_gradient and self.model_compression)

    def get_embedding_from_word(self, word):
        token_id = self.tokenizer.convert_tokens_to_ids([word])[0]
        return self.embedding_mat[token_id, :]

    def get_baseline(self, baseline, input_mask):
        actual_seq_length = tf.reduce_sum(input_mask)
        if baseline == 'zero':
            embedding_baseline = tf.zeros(self.hidden_size)
        elif baseline == 'mask' or baseline == 'mask_i' or 'mask' in baseline:
            embedding_baseline = self.mask_embedding
        elif baseline == 'pad' or 'pad' in baseline:
            embedding_baseline = self.pad_embedding
        else:
            embedding_baseline = self.get_embedding_from_word(baseline)
        embedding_baseline = tf.expand_dims(embedding_baseline, 0)
        if 'cls' in baseline and 'sep' in baseline:
            return tf.concat([
                tf.expand_dims(self.cls_embedding, 0),
                tf.repeat(embedding_baseline, actual_seq_length - 2, axis=0),
                tf.expand_dims(self.sep_embedding, 0),
                tf.repeat(embedding_baseline,
                          self.seq_length - actual_seq_length,
                          axis=0),
            ], 0)

        elif 'cls' in baseline:

            return tf.concat([
                tf.expand_dims(self.cls_embedding, 0),
                tf.repeat(embedding_baseline, actual_seq_length - 1, axis=0),
                tf.repeat(embedding_baseline,
                          self.seq_length - actual_seq_length,
                          axis=0),
            ], 0)
        elif 'sep' in baseline:
            return tf.concat([
                tf.repeat(embedding_baseline, actual_seq_length - 1, axis=0),
                tf.expand_dims(self.sep_embedding, 0),
                tf.repeat(embedding_baseline,
                          self.seq_length - actual_seq_length,
                          axis=0),
            ], 0)
        else:
            return tf.repeat(embedding_baseline, self.seq_length, axis=0)

    def get_tensor_input_batch(self, examples):
        batch_size = len(examples)
        make_constant = lambda m: tf.constant(m, tf.int32)
        input_word_ids = make_constant(
            np.vstack([e.input_ids for e in examples]))
        input_mask = make_constant(np.vstack([e.input_mask for e in examples]))
        if self.decoder_type == 'mlm':
            masked_lm_positions = make_constant(
                np.vstack([[e.target_idx] for e in examples]))
        else:
            masked_lm_positions = None
        segment_ids = make_constant(np.vstack([e.segment_ids for e in examples
                                              ]))
        index = np.vstack([e.word_ids for e in examples]).astype(np.int32)
        index = np.vstack((np.arange(batch_size).repeat(2),
                           index.reshape(1, -1))).T.reshape(len(index), 2,
                                                            2).astype(np.int32)
        return input_word_ids, input_mask, segment_ids, index, masked_lm_positions

    def construct_qoi(self, word_ids, res):
        """
        docstring
        """
        index = np.array(word_ids).astype(np.int32)
        index = tf.constant(
            np.hstack([
                np.array([[r] * len(word_ids), index]) for r in range(res)
            ]).T.reshape(res, len(word_ids), 2).astype(np.int32))
        return index

    def get_tensor_input(self, e, res=1, agg_verb=False):
        make_constant = lambda m: tf.repeat(
            tf.constant([m], tf.int32), res, axis=0)
        input_word_ids = tf.constant([e.input_ids])
        segment_ids = make_constant(e.segment_ids)
        input_mask = make_constant(e.input_mask)
        qoi_index = self.construct_qoi(e.word_ids, res)
        if self.decoder_type == 'mlm':
            masked_lm_positions = make_constant([e.target_idx])
        else:
            masked_lm_positions = None
        return input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions

    def get_alphas(self, res):
        return tf.linspace(0.0, 1.0, res)

    def get_path_embeddings(self, input_ids, embedding_baseline, alphas):
        # Expand dimensions for vectorized computation of interpolations.
        input_x = self.embedding_model(input_ids)
        alphas_x = alphas[:, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(embedding_baseline, axis=0)
        # print(baseline_x.shape, alphas_x.shape, input_x.shape)
        # print(alphas_x * (input_x - baseline_x))
        path_embeddings = baseline_x + alphas_x * (input_x - baseline_x)
        return path_embeddings

    def get_index_node_mask(self,
                            index1,
                            index2,
                            ei,
                            top_indices,
                            top_attention_indices,
                            incl_skip_nodes,
                            res,
                            start=0,
                            end=1,
                            endpoint=True):
        # print('ti', top_indices)
        if index1 is None:
            #if no word index is given, assume end to end calculation.
            assert top_attention_indices is None
            assert top_indices is None
        if top_indices is not None:
            # print('top indice not None')
            emb_node_mask = self.get_emb_node_mask(index1, index2, ei,
                                                   top_indices, res)
        else:
            emb_node_mask = tf.ones((res, self.num_hidden_layers + 1,
                                     self.seq_length, self.hidden_size))
        if top_attention_indices is not None:
            # print('top attn indice not None')

            attn_node_mask = self.get_attn_node_mask(index1, index2, ei,
                                                     top_attention_indices, res)
        else:
            attn_node_mask = tf.ones(
                (res, self.num_hidden_layers, self.seq_length,
                 self.num_attention_heads,
                 int(self.hidden_size / self.num_attention_heads)))
        if incl_skip_nodes is None:
            # print('top incl None')

            incl_skip_node = tf.ones((1, self.num_hidden_layers))
        else:
            incl_skip_node = tf.constant(incl_skip_nodes[ei,
                                                         index2, :][None, :])
        return emb_node_mask, attn_node_mask, incl_skip_node

    def get_emb_node_mask(self, index1, index2, ei, top_indices, res):
        """
        top_indices: np.ones((len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length))
        """
        top_indices_e = np.expand_dims(top_indices[ei], 0)
        emb_node_mask = np.expand_dims(
            np.concatenate(
                (self.indice_array[:, :,
                                   index1, :], top_indices_e[:, index2, :, :]),
                axis=1).repeat(res, 0), -1).repeat(self.hidden_size, -1)
        emb_node_mask = tf.constant(emb_node_mask)
        return emb_node_mask

    def get_attn_node_mask(self, index1, index2, ei, top_attention_indices,
                           res):
        """
        top_attention_indices: np.ones((len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length), len(num_of_heads))
                                   
        """
        top_indices_e = np.expand_dims(top_attention_indices[ei], 0)
        attn_node_mask = np.expand_dims(
            top_indices_e[:, index2, :, :, :].repeat(res, 0),
            -1).repeat(self.attention_size, -1)
        attn_node_mask = tf.constant(attn_node_mask)
        return attn_node_mask

    def tf_forward_pass(
        self,
        alphas,
        input_word_ids,
        embedding_baseline,
        input_mask,
        segment_ids,
        qoi_index,
        masked_lm_positions=None,
        emb_node_mask=None,
        attn_node_mask=None,
        incl_skip_node=None,
        emb_node_mask_rep=None,
        attn_node_mask_rep=None,
        incl_skip_mask_rep=None,
    ):
        return tf.function(self.forward_pass)(
            alphas, input_word_ids, embedding_baseline, input_mask, segment_ids,
            qoi_index, masked_lm_positions, emb_node_mask, attn_node_mask,
            incl_skip_node, emb_node_mask_rep, attn_node_mask_rep,
            incl_skip_mask_rep)

    # @tf.function
    def forward_pass(
        self,
        alphas,
        input_word_ids,
        embedding_baseline,
        input_mask,
        segment_ids,
        qoi_index,
        masked_lm_positions=None,
        emb_node_mask=None,
        attn_node_mask=None,
        incl_skip_node=None,
        emb_node_mask_rep=None,
        attn_node_mask_rep=None,
        incl_skip_mask_rep=None,
    ):
        path_embeddings = self.get_path_embeddings(input_word_ids,
                                                   embedding_baseline, alphas)
        if self.use_stop_gradient:
            sequence_output, attention_output, attention_probs, cls_output = self.transformer_encoder(
                [
                    input_mask, segment_ids, path_embeddings, emb_node_mask,
                    attn_node_mask, incl_skip_node
                ])
        elif self.model_compression:
            sequence_output, attention_output, attention_probs, cls_output = self.transformer_encoder(
                [
                    input_mask, segment_ids, path_embeddings, emb_node_mask,
                    attn_node_mask, incl_skip_node, emb_node_mask_rep,
                    attn_node_mask_rep, incl_skip_mask_rep
                ])
        else:
            sequence_output, attention_output, attention_probs, cls_output = self.transformer_encoder(
                [input_mask, segment_ids, path_embeddings])
        if self.decoder_type == 'cls':
            decoder_output = self.decoder_model([cls_output])
        elif self.decoder_type == 'mlm':
            decoder_output = tf.squeeze(
                self.decoder_model([sequence_output[-1], masked_lm_positions]),
                1)
        decoder_output_difference = tf.gather_nd(decoder_output, qoi_index)
        return decoder_output_difference, decoder_output, attention_output, sequence_output, attention_probs

    @tf.function
    def get_reverse_gradient(self, alphas, input_word_ids, embedding_baseline,
                             input_mask, segment_ids, use_attention_output,
                             layer, qoi_index, **kwargs):
        with tf.GradientTape(watch_accessed_variables=False,
                             persistent=False) as tape:
            tape.watch(alphas)
            decoder_output_difference, _, attention_output, sequence_output, _ = self.forward_pass(
                alphas, input_word_ids, embedding_baseline, input_mask,
                segment_ids, qoi_index, **kwargs)
            decoder_output_difference_qoi = tf.reduce_sum(
                decoder_output_difference[:, 0] -
                decoder_output_difference[:, 1])
            lm_diff = sequence_output[0][-1] - embedding_baseline
            if layer == 'alpha':
                return tape.gradient(decoder_output_difference_qoi, alphas)
            if layer is not None:
                attention_output = attention_output[layer]
                sequence_output = sequence_output[layer + 2]

            if use_attention_output:
                grad = tape.gradient(decoder_output_difference_qoi,
                                     attention_output)
            else:
                grad = tape.gradient(decoder_output_difference_qoi,
                                     sequence_output)
            if layer is None:
                grad = tf.stack(grad, axis=1)
            # grad = (grad[:-1] + grad[1:]) / tf.constant(2.0)

        return grad, decoder_output_difference, lm_diff

    @tf.function
    def get_forward_gradient(self, alphas, input_word_ids, embedding_baseline,
                             input_mask, segment_ids, use_attention_output,
                             layer, qoi_index, **kwargs):
        with tf.autodiff.ForwardAccumulator(
                primals=alphas, tangents=tf.ones_like(alphas)) as acc:
            _, _, attention_output, sequence_output, _ = self.forward_pass(
                alphas, input_word_ids, embedding_baseline, input_mask,
                segment_ids, qoi_index, **kwargs)
            if layer is not None:
                attention_output = attention_output[layer]
                sequence_output = sequence_output[layer + 2]

            if use_attention_output:
                jvps = acc.jvp(attention_output)
            else:
                jvps = acc.jvp(sequence_output)
            if layer is None:
                jvps = tf.stack(jvps, axis=1)
            return jvps

    def get_reverse_influence_optimality_test(
        self,
        examples,
        res,
        indices,
        num_random=100,
        #   top_indices=None,
        #   top_attention_indices=None,
        #   incl_skip_nodes=None,
        use_attention_output=False,
        reverse_qoi=None,
        baseline='zero',
        fix_top_indice=None,
        fix_attention_indice=None):
        """[summary]

        Args:
            examples ([type]): [description]
            res ([type]): [description]
            indices ([type]): word indices to calculate influence of
            layer ([type], optional): [description]. Defaults to None.
            top_indices ([type], optional): [description]. Defaults to None.
            top_attention_indices ([type], optional): [description]. Defaults to None.
            incl_skip_nodes ([type], optional): [description]. Defaults to None.
            use_attention_output (bool, optional): [description]. Defaults to False.
            reverse_qoi ([type], optional): [description]. Defaults to None.
            baseline (str, optional): [description]. Defaults to 'zero'.

        Returns:
            [type]: [description]
        """

        assert self.use_stop_gradient
        i2, i1 = np.meshgrid(np.arange(len(indices)), np.arange(len(examples)))
        ## all attention indices mask default 1 (allow flow of all heads)

        # if not use_attention_output:
        top_attention_indices = np.ones(
            (num_random, len(examples), len(indices), self.num_hidden_layers,
             self.seq_length, self.num_attention_heads))
        ## in the beginning, start with including all skip connections.
        incl_skip_nodes = np.ones(
            (num_random, len(examples), len(indices), self.num_hidden_layers))
        # else:

        #     top_attention_indices = np.zeros(
        #         (num_random, len(examples), len(indices),
        #          self.num_hidden_layers, self.seq_length,
        #          self.num_attention_heads))
        #     ## in the beginning, start with including all skip connections.
        #     incl_skip_nodes = np.zeros((num_random, len(examples), len(indices),
        #                                 self.num_hidden_layers))
        if fix_top_indice is None:
            ### getting embedding level path
            top_indices = np.zeros((num_random, len(examples), len(indices),
                                    self.num_hidden_layers, self.seq_length))
            reverse_qoi = np.zeros((len(examples), len(indices)))
        else:
            fix_top_indice = fix_top_indice.astype(int)
            top_indices = np.ones((num_random, len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length))
            # if use_attention_output:

        if use_attention_output:
            layer_range = np.arange(0, self.num_hidden_layers)
        else:
            layer_range = np.arange(0, self.num_hidden_layers)
        for r in np.arange(num_random):
            for l in layer_range:
                top_indices[r, :, :, l, :] = 0
                top_indices[r, i1, i2, l, fix_top_indice[r, l, :, :, 0]] = 1
                if use_attention_output:
                    top_attention_indices[r, :, :, l, :, :] = 0
                    # top_attention_indices[r, i1, i2, l,
                    #                       fix_top_indice[r, l, :, :, 0], :] = 1
                    for ie in range(len(examples)):
                        for ii in range(len(indices)):
                            # print(r, ie, ii, l, fix_attention_indice[r, l, ie,
                            #                                          ii])

                            if fix_attention_indice[r, l, ie, ii] == -1:
                                incl_skip_nodes[r, ie, ii, l] = 1
                                top_attention_indices[r, ie, ii, l,
                                                      fix_top_indice[r, l, ie,
                                                                     ii,
                                                                     0], :] = 0
                            else:
                                incl_skip_nodes[r, ie, ii, l] = 0
                                top_attention_indices[
                                    r, ie, ii, l, fix_top_indice[r, l, ie, ii,
                                                                 0],
                                    fix_attention_indice[r, l, ie, ii]] = 1
                # top_attention_indices[:, :, l, :, :] = 0
                # top_attention_indices[i1, i2, l, fix_top_indice[l, :, :,
                #                                                 0], :] = 1
        # print(top_attention_indices[0, 0, 0], top_attention_indices.shape)
        # print(incl_skip_nodes, incl_skip_nodes.shape)
        # print(top_indices[0, 0, 0])
        all_grads = []
        for ei, e in tqdm(enumerate(examples), position=0, leave=True):
            # e_grad = []
            input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions = self.get_tensor_input(
                e,
                res + 1,
            )
            embedding_baseline = self.get_baseline(baseline, e.input_mask)
            alphas = self.get_alphas(res + 1)
            for index2, index1 in enumerate(indices):

                if index1 == None or e.input_mask[index1] != 0:
                    # print(index1)
                    r_grad = []
                    for t in range(num_random):
                        # for t in range(len(top_indices)):

                        emb_node_mask, attn_node_mask, incl_skip_node = self.get_index_node_mask(
                            index1, index2, ei, top_indices[t],
                            top_attention_indices[t], incl_skip_nodes[t],
                            res + 1)
                        grad = self.get_reverse_gradient(
                            alphas,
                            input_word_ids,
                            embedding_baseline,
                            input_mask,
                            segment_ids,
                            use_attention_output,
                            'alpha',
                            qoi_index,
                            emb_node_mask=emb_node_mask,
                            attn_node_mask=attn_node_mask,
                            incl_skip_node=incl_skip_node,
                            masked_lm_positions=masked_lm_positions,
                        )
                        if reverse_qoi[ei, index2]:
                            grad = -grad
                        grad = grad.numpy().mean()
                        # gradr, _, _ = self.get_reverse_gradient(
                        #     alphas,
                        #     input_word_ids,
                        #     embedding_baseline,
                        #     input_mask,
                        #     segment_ids,
                        #     use_attention_output,
                        #     5,
                        #     qoi_index,
                        #     emb_node_mask=emb_node_mask,
                        #     attn_node_mask=attn_node_mask,
                        #     incl_skip_node=incl_skip_node,
                        #     masked_lm_positions=masked_lm_positions,
                        # )

                        # gradf = self.get_forward_gradient(
                        #     alphas,
                        #     input_word_ids,
                        #     embedding_baseline,
                        #     input_mask,
                        #     segment_ids,
                        #     use_attention_output,
                        #     5,
                        #     qoi_index,
                        #     emb_node_mask=emb_node_mask,
                        #     attn_node_mask=attn_node_mask,
                        #     incl_skip_node=incl_skip_node,
                        #     masked_lm_positions=masked_lm_positions,
                        # )
                        # print(gradr.shape, gradf.shape)
                        # grad = (gradf * gradr).numpy()
                        # grad = grad.sum(-1).mean(0)
                        # print(grad.sum())
                        # if reverse_qoi[ei, index2]:
                        #     grad = -grad
                        # grad = grad.
                        # print(grad.mean())
                        # else:
                        ##padding
                        # print(index2, index1)
                        # if use_attention_output:
                        #     grad = np.zeros(
                        #         (res + 1, self.seq_length, self.num_hidden_layers,
                        #          self.attention_size))
                        # else:
                        #     grad = np.zeros(
                        #         (res + 1, self.seq_length, self.hidden_size))

                        r_grad.append(grad)
                # e_grad.append(r_grad)
                all_grads.append(r_grad)
        return np.array(all_grads)

    def get_reverse_influence(self,
                              examples,
                              res,
                              indices,
                              layer=None,
                              top_indices=None,
                              top_attention_indices=None,
                              incl_skip_nodes=None,
                              use_attention_output=False,
                              reverse_qoi=None,
                              baseline='zero'):
        """[summary]

        Args:
            examples ([type]): [description]
            res ([type]): [description]
            indices ([type]): word indices to calculate influence of
            layer ([type], optional): [description]. Defaults to None.
            top_indices ([type], optional): [description]. Defaults to None.
            top_attention_indices ([type], optional): [description]. Defaults to None.
            incl_skip_nodes ([type], optional): [description]. Defaults to None.
            use_attention_output (bool, optional): [description]. Defaults to False.
            reverse_qoi ([type], optional): [description]. Defaults to None.
            baseline (str, optional): [description]. Defaults to 'zero'.

        Returns:
            [type]: [description]
        """
        assert self.use_stop_gradient
        if not isinstance(reverse_qoi, np.ndarray):
            if reverse_qoi == True:
                reverse_qoi = np.ones((len(examples), len(indices)))
            elif reverse_qoi == False or reverse_qoi is None:
                reverse_qoi = np.zeros((len(examples), len(indices)))
        all_grads = []
        # use_attention_output = tf.constant(use_attention_output)
        for ei, e in tqdm(enumerate(examples), position=0, leave=True):

            e_grad = []
            input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions = self.get_tensor_input(
                e,
                res + 1,
            )
            embedding_baseline = self.get_baseline(baseline, e.input_mask)
            alphas = self.get_alphas(res + 1)
            for index2, index1 in enumerate(indices):
                if index1 == None or e.input_mask[index1] != 0:

                    emb_node_mask, attn_node_mask, incl_skip_node = self.get_index_node_mask(
                        index1, index2, ei, top_indices, top_attention_indices,
                        incl_skip_nodes, res + 1)
                    grad, _, _ = self.get_reverse_gradient(
                        alphas,
                        input_word_ids,
                        embedding_baseline,
                        input_mask,
                        segment_ids,
                        use_attention_output,
                        layer,
                        qoi_index,
                        emb_node_mask=emb_node_mask,
                        attn_node_mask=attn_node_mask,
                        incl_skip_node=incl_skip_node,
                        masked_lm_positions=masked_lm_positions,
                    )
                    if reverse_qoi[ei, index2]:
                        grad = -grad
                    grad = grad.numpy()
                else:
                    ##padding
                    # print(index2, index1)
                    if use_attention_output:
                        grad = np.zeros(
                            (res + 1, self.seq_length, self.num_hidden_layers,
                             self.attention_size))
                    else:
                        grad = np.zeros(
                            (res + 1, self.seq_length, self.hidden_size))

                e_grad.append(grad)
            all_grads.append(e_grad)
        return np.array(all_grads)

    def get_forward_influence(self,
                              examples,
                              res,
                              indices,
                              layer=None,
                              top_indices=None,
                              top_attention_indices=None,
                              incl_skip_nodes=None,
                              use_attention_output=False,
                              baseline='zero'):
        assert self.use_stop_gradient
        all_forward_grads = []
        for ei, e in tqdm(enumerate(examples), position=0, leave=True):
            forward_grads = []
            input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions = self.get_tensor_input(
                e,
                res + 1,
            )
            embedding_baseline = self.get_baseline(baseline, e.input_mask)
            alphas = self.get_alphas(res + 1)
            for index2, index1 in enumerate(indices):
                if index1 == None or e.input_mask[index1] != 0:

                    emb_node_mask, attn_node_mask, incl_skip_node = self.get_index_node_mask(
                        index1, index2, ei, top_indices, top_attention_indices,
                        incl_skip_nodes, res + 1)
                    forward_grad = self.get_forward_gradient(
                        alphas,
                        input_word_ids,
                        embedding_baseline,
                        input_mask,
                        segment_ids,
                        use_attention_output,
                        layer,
                        qoi_index,
                        emb_node_mask=emb_node_mask,
                        attn_node_mask=attn_node_mask,
                        incl_skip_node=incl_skip_node,
                        masked_lm_positions=masked_lm_positions,
                    )
                    forward_grad = forward_grad.numpy()
                    # print(forward_grad.shape)
                else:
                    if use_attention_output:
                        forward_grad = np.zeros(
                            (res + 1, self.seq_length, self.num_hidden_layers,
                             self.attention_size))
                    else:
                        forward_grad = np.zeros(
                            (res + 1, self.seq_length, self.hidden_size))
                forward_grads.append(forward_grad)
            all_forward_grads.append(forward_grads)
        return np.array(all_forward_grads)

    def get_lm_prob_batch(self,
                          examples,
                          batch_size,
                          return_attention_prob=False,
                          return_decoder_output=False,
                          return_decoder_output_mask=False):
        assert not self.use_stop_gradient and not self.model_compression
        decoder_outputs = np.zeros((0, 2))
        category_features = []
        attention_prob_lst = []
        sequence_output_all = np.zeros(
            (0, self.num_hidden_layers + 1, self.seq_length, self.hidden_size))
        attn_output_all = np.zeros(
            (0, self.num_hidden_layers, self.seq_length,
             self.num_attention_heads, self.attention_size))
        for ei, e in tqdm(enumerate(examples_in_batches(examples, batch_size)),
                          position=0,
                          leave=True):
            actual_batch_size = len(e)
            input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions = self.get_tensor_input_batch(
                e)

            alphas = tf.ones(actual_batch_size, self.precision)
            embedding_baseline = tf.zeros((self.seq_length, self.hidden_size))
            decoder_output_difference, decoder_output, attention_output, sequence_output, attention_probs = self.forward_pass(
                alphas,
                input_word_ids,
                embedding_baseline,
                input_mask,
                segment_ids,
                qoi_index,
                masked_lm_positions=masked_lm_positions)

            if not return_decoder_output and not return_decoder_output_mask:
                sequence_output_all = np.concatenate(
                    (sequence_output_all, tf.stack(sequence_output[1:],
                                                   axis=1).numpy()), 0)
                # print(tf.stack(attention_output, axis=1).numpy().shape)
                attn_output_all = np.concatenate(
                    (attn_output_all, tf.stack(attention_output,
                                               axis=1).numpy()), 0)

            if return_attention_prob:
                attention_probs = np.array([a.numpy() for a in attention_probs])
                attention_probs = np.swapaxes(attention_probs, 0, 1)
                attention_prob_lst.append(attention_probs)

            if return_decoder_output_mask:
                assert len(examples) == 1
                assert actual_batch_size == 1
                return np.array(tf.squeeze(decoder_output_difference).numpy())
            # print(decoder_output.shape, decoder_outputs.shape)
            decoder_outputs = np.vstack(
                (decoder_outputs, decoder_output_difference))
            if self.decoder_type == 'mlm':
                category_features.extend([
                    (eii.features[0], eii.features[1]) for eii in e
                ])
        if return_decoder_output:
            return np.array(decoder_outputs)

        if return_attention_prob:
            attention_prob_result = np.concatenate(attention_prob_lst)
        else:
            attention_prob_result = None
        return np.array(
            decoder_outputs
        ), category_features, attention_prob_result, sequence_output_all, attn_output_all

    def get_e2e_influence(self,
                          examples,
                          res,
                          start=0,
                          end=1,
                          endpoint=True,
                          return_all_resolutions=False,
                          return_wrt2_embedding=False,
                          baseline='zero',
                          agg_verb=False,
                          ig=True):
        assert not self.use_stop_gradient

        all_grad_is = []
        decoder_output_differences = []

        for ei, e in tqdm(enumerate(examples)):
            # if ei % 100 == 0:
            #     print(ei)
            grad_is = []
            decoder_output_difference_lst = []
            input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions = self.get_tensor_input(
                e,
                res + 1,
            )
            embedding_baseline = self.get_baseline(baseline, e.input_mask)
            alphas = self.get_alphas(res + 1)
            grad_i, decoder_output_difference, lm_diff = self.get_reverse_gradient(
                alphas,
                input_word_ids,
                embedding_baseline,
                input_mask,
                segment_ids,
                use_attention_output=False,
                layer=-2,
                qoi_index=qoi_index,
                masked_lm_positions=masked_lm_positions)
            grad_i = tf.reduce_sum((grad_i * lm_diff), -1).numpy()
            grad_is.append(grad_i)
            all_grad_is.append(grad_is)
            decoder_output_difference_lst.append(
                decoder_output_difference.numpy())
            decoder_output_differences.append(decoder_output_difference_lst)
        return np.array(all_grad_is), np.array(decoder_output_differences)

    def get_model_compression(self,
                              examples,
                              batch_size,
                              ind_heatmap_e=None,
                              sequence_output_e=None,
                              attn_heatmap_e=None,
                              attn_output_e=None,
                              ind_skip_e=None):

        assert self.model_compression and not self.use_stop_gradient
        decoder_outputs = np.zeros((0, 2))
        if ind_heatmap_e is None:
            ind_heatmap_e = np.ones((len(examples), self.num_hidden_layers + 1,
                                     self.seq_length, self.hidden_size))
        if sequence_output_e is None:
            sequence_output_e = np.zeros(
                (self.num_hidden_layers + 1, self.seq_length, self.hidden_size))
        if attn_output_e is None:
            attn_output_e = np.zeros(
                (self.num_hidden_layers, self.seq_length,
                 self.num_attention_heads, self.attention_size))
        if attn_heatmap_e is None:
            attn_heatmap_e = np.ones(
                (len(examples), self.num_hidden_layers, self.seq_length,
                 self.num_attention_heads, self.attention_size))
        if ind_skip_e is None:
            ind_skip_e = np.ones((len(examples), self.num_hidden_layers,
                                  self.seq_length, self.hidden_size))
        # for ei, e in tqdm(enumerate(examples_in_batches(examples, batch_size))):
        for i in tqdm(range(1 + ((len(examples) - 1) // batch_size)),
                      position=0,
                      leave=True):
            e = examples[i * batch_size:(i + 1) * batch_size]
            actual_batch_size = len(e)
            input_word_ids, input_mask, segment_ids, qoi_index, masked_lm_positions = self.get_tensor_input_batch(
                e)

            alphas = tf.ones(actual_batch_size, self.precision)
            embedding_baseline = tf.zeros((self.seq_length, self.hidden_size))
            decoder_output_difference, _, _, _, _ = self.forward_pass(
                alphas,
                input_word_ids,
                embedding_baseline,
                input_mask,
                segment_ids,
                qoi_index,
                emb_node_mask_rep=tf.constant(
                    sequence_output_e[None, ...].repeat(actual_batch_size, 0)),
                attn_node_mask_rep=tf.constant(attn_output_e[None, ...].repeat(
                    actual_batch_size, 0)),
                incl_skip_mask_rep=tf.constant(
                    ind_skip_e[i * batch_size:(i + 1) * batch_size]),
                emb_node_mask=tf.constant(ind_heatmap_e[i * batch_size:(i + 1) *
                                                        batch_size]),
                attn_node_mask=tf.constant(
                    attn_heatmap_e[i * batch_size:(i + 1) * batch_size]),
                incl_skip_node=tf.constant(np.zeros(
                    (1, self.num_hidden_layers))),
                masked_lm_positions=masked_lm_positions,
            )

            # input_mask, segment_ids, path_embeddings, emb_node_mask,
            #         attn_node_mask, incl_skip_node, emb_node_mask_rep,
            #         attn_node_mask_rep, incl_skip_mask_rep

            # sequence_output, attention_output, _, _ = self.transformer_encoder([
            #     input_word_ids,
            #     input_mask,
            #     segment_ids,
            #     embedding_end,
            #     embedding_baseline,
            #     alpha,
            #     tf.constant(ind_heatmap_e[i * batch_size:(i + 1) * batch_size]),
            #     tf.constant(attn_heatmap_e[i * batch_size:(i + 1) *
            #                                batch_size]),
            #     tf.constant(np.zeros((1, self.num_hidden_layers))),
            #     tf.constant(sequence_output_e[None,
            #                                   ...].repeat(actual_batch_size,
            #                                               0)),
            #     tf.constant(attn_output_e[None,
            #                               ...].repeat(actual_batch_size, 0)),
            #     tf.constant(ind_skip_e[i * batch_size:(i + 1) * batch_size]),
            # ])

            decoder_outputs = np.vstack(
                (decoder_outputs, decoder_output_difference))
        return np.array(decoder_outputs)

    def get_greedy_influence_test(self,
                                  examples,
                                  res,
                                  indices,
                                  threshold=0,
                                  use_attention_output=False,
                                  reverse_qoi=None,
                                  baseline='zero',
                                  fix_top_indice=None):
        i2, i1 = np.meshgrid(np.arange(len(indices)), np.arange(len(examples)))
        ## all attention indices mask default 1 (allow flow of all heads)
        top_attention_indices = np.ones(
            (len(examples), len(indices), self.num_hidden_layers,
             self.seq_length, self.num_attention_heads))
        ## in the beginning, start with including all skip connections.
        incl_skip_nodes = np.ones(
            (len(examples), len(indices), self.num_hidden_layers))

        if fix_top_indice is None:
            ### getting embedding level path
            top_indices = np.ones((len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length))
            assert reverse_qoi is None
            reverse_qoi = np.zeros((len(examples), len(indices)))
        else:
            assert threshold == 0  ## for attention-level path, assume picking only 1 node per layer
            fix_top_indice = fix_top_indice.astype(int)
            top_indices = np.ones((len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length))
            for l in np.arange(0, self.num_hidden_layers):
                top_indices[:, :, l, :] = 0
                top_indices[i1, i2, l, fix_top_indice[l, :, :, 0]] = 1
                top_attention_indices[:, :, l, :, :] = 0
                top_attention_indices[i1, i2, l, fix_top_indice[l, :, :,
                                                                0], :] = 1
            reverse_qoi_idx = np.where(reverse_qoi == 1)

        expanding_indices = []
        combs = []
        if fix_top_indice is None:
            # if getting embedding level path, last layer's node is fixed (to mask)
            layer_range = np.arange(0, self.num_hidden_layers - 1)
            # layer_range = np.arange(1)

        else:
            layer_range = np.arange(0, self.num_hidden_layers)
        for layer in [0]:
            print('=' * 50)
            print('layer', layer)
            fwd = self.get_forward_influence(
                examples,
                res,
                indices,
                layer,
                top_indices,
                top_attention_indices=top_attention_indices,
                incl_skip_nodes=incl_skip_nodes,
                use_attention_output=use_attention_output,
                baseline=baseline)
            # print(fwd.shape)
            # print('backward influence')
            rvs = self.get_reverse_influence(
                examples,
                res,
                indices,
                layer,
                top_indices,
                top_attention_indices=top_attention_indices,
                incl_skip_nodes=incl_skip_nodes,
                use_attention_output=use_attention_output,
                reverse_qoi=reverse_qoi,
                baseline=baseline)
            comb = fwd * rvs  # (E, 10, 50, 11, 512)

            comb = np.mean(comb.sum(-1), axis=2)

            if fix_top_indice is None or not use_attention_output:
                # pass
                # if layer ==1 and
                if layer == 0:
                    # modify reverse qoi based on layer
                    neg_ind = np.where(comb.sum(-1) < 0)
                    reverse_qoi[neg_ind[0], neg_ind[1]] = 1
                    comb[neg_ind[0], neg_ind[1], :] *= -1
                # argsort_index = np.argsort(-comb, axis=-1)
                # expanding_indices.append(argsort_index[:, :, :threshold + 1])

                # print(comb.shape)
                combs.append(comb)

        for layer in [1]:
            print('=' * 50)
            print('layer', layer)
            for tmp_index in range(self.seq_length):
                top_indices[:, :, 0, :] = 0
                top_indices[i1, i2, 0, tmp_index] = 1
                fwd = self.get_forward_influence(
                    examples,
                    res,
                    indices,
                    layer,
                    top_indices,
                    top_attention_indices=top_attention_indices,
                    incl_skip_nodes=incl_skip_nodes,
                    use_attention_output=use_attention_output,
                    baseline=baseline)
                # print(fwd.shape)
                # print('backward influence')
                rvs = self.get_reverse_influence(
                    examples,
                    res,
                    indices,
                    layer,
                    top_indices,
                    top_attention_indices=top_attention_indices,
                    incl_skip_nodes=incl_skip_nodes,
                    use_attention_output=use_attention_output,
                    reverse_qoi=reverse_qoi,
                    baseline=baseline)
                comb = fwd * rvs  # (E, 10, 50, 11, 512)

                comb = np.mean(comb.sum(-1), axis=2)

                combs.append(comb)
        np.testing.assert_almost_equal(combs[0].sum(-1),
                                       np.array(combs[1:]).sum(0).sum(-1),
                                       decimal=5,
                                       err_msg='',
                                       verbose=True)

        return combs

    def get_greedy_influence(self,
                             examples,
                             res,
                             indices,
                             threshold=0,
                             use_attention_output=False,
                             reverse_qoi=None,
                             baseline='zero',
                             fix_top_indice=None):
        """
        
        """
        i2, i1 = np.meshgrid(np.arange(len(indices)), np.arange(len(examples)))
        ## all attention indices mask default 1 (allow flow of all heads)
        top_attention_indices = np.ones(
            (len(examples), len(indices), self.num_hidden_layers,
             self.seq_length, self.num_attention_heads))
        ## in the beginning, start with including all skip connections.
        incl_skip_nodes = np.ones(
            (len(examples), len(indices), self.num_hidden_layers))

        if fix_top_indice is None:
            ### getting embedding level path
            top_indices = np.ones((len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length))
            assert reverse_qoi is None
            reverse_qoi = np.zeros((len(examples), len(indices)))
        else:
            assert threshold == 0  ## for attention-level path, assume picking only 1 node per layer
            fix_top_indice = fix_top_indice.astype(int)
            top_indices = np.ones((len(examples), len(indices),
                                   self.num_hidden_layers, self.seq_length))
            for l in np.arange(0, self.num_hidden_layers):
                top_indices[:, :, l, :] = 0
                top_indices[i1, i2, l, fix_top_indice[l, :, :, 0]] = 1
                top_attention_indices[:, :, l, :, :] = 0
                top_attention_indices[i1, i2, l, fix_top_indice[l, :, :,
                                                                0], :] = 1
            reverse_qoi_idx = np.where(reverse_qoi == 1)

        expanding_indices = []
        combs = []
        if fix_top_indice is None:
            # if getting embedding level path, last layer's node is fixed (to mask)
            layer_range = np.arange(0, self.num_hidden_layers - 1)
            # layer_range = np.arange(1)

        else:
            layer_range = np.arange(0, self.num_hidden_layers)
        for layer in layer_range:
            print('=' * 50)
            print('layer', layer)
            fwd = self.get_forward_influence(
                examples,
                res,
                indices,
                layer,
                top_indices,
                top_attention_indices=top_attention_indices,
                incl_skip_nodes=incl_skip_nodes,
                use_attention_output=use_attention_output,
                baseline=baseline)
            # print(fwd.shape)
            # print('backward influence')
            rvs = self.get_reverse_influence(
                examples,
                res,
                indices,
                layer,
                top_indices,
                top_attention_indices=top_attention_indices,
                incl_skip_nodes=incl_skip_nodes,
                use_attention_output=use_attention_output,
                reverse_qoi=reverse_qoi,
                baseline=baseline)
            comb = fwd * rvs  # (E, 10, 50, 11, 512)

            comb = np.mean(comb.sum(-1), axis=2)

            if fix_top_indice is None or not use_attention_output:
                # pass
                # if layer ==1 and
                if layer == 0:
                    # modify reverse qoi based on layer
                    neg_ind = np.where(comb.sum(-1) < 0)
                    reverse_qoi[neg_ind[0], neg_ind[1]] = 1
                    comb[neg_ind[0], neg_ind[1], :] *= -1
                argsort_index = np.argsort(-comb, axis=-1)
                expanding_indices.append(argsort_index[:, :, :threshold + 1])
                top_indices[:, :, layer, :] = 0
                for t in range(threshold + 1):
                    top_indices[i1, i2, layer, argsort_index[:, :, t]] = 1
                # print(comb.shape)
                combs.append(comb)
            else:
                # pass
                comb_attention = comb[i1, i2, fix_top_indice[layer, :, :, 0], :]
                ##skip connection
                top_attention_indices[:, :, layer, :, :] = 0
                fwd = self.get_forward_influence(
                    examples,
                    res,
                    indices,
                    layer,
                    top_indices,
                    top_attention_indices=top_attention_indices,
                    incl_skip_nodes=incl_skip_nodes,
                    use_attention_output=False,
                    baseline=baseline)
                rvs = self.get_reverse_influence(
                    examples,
                    res,
                    indices,
                    layer,
                    top_indices,
                    top_attention_indices=top_attention_indices,
                    incl_skip_nodes=incl_skip_nodes,
                    use_attention_output=False,
                    reverse_qoi=reverse_qoi,
                    baseline=baseline)
                comb_rep = fwd * rvs
                comb_rep = np.mean(comb_rep.sum(-1), axis=2)
                comb_skip = comb_rep[i1, i2, fix_top_indice[layer, :, :, 0]]

                # print(comb_attention.sum(-1) + comb_skip)

                argsort_index = np.argsort(-comb_attention, axis=-1)[:, :, 0]
                max_comb_attention = np.max(comb_attention, -1)
                if layer >= self.num_hidden_layers - 2:
                    print(comb_skip, max_comb_attention)
                attn_win_index = np.where(max_comb_attention > comb_skip)
                skip_win_index = np.where(max_comb_attention <= comb_skip)
                incl_skip_nodes[
                    attn_win_index[0], attn_win_index[1],
                    layer] = 0  # if a path through an attention head is better than skip connection, do not keep the skip connection.
                # top_attention_indices[:, :, layer, :, :] = 0
                top_attention_indices[i1, i2, layer, fix_top_indice[layer, :, :,
                                                                    0],
                                      argsort_index] = 1
                top_attention_indices[
                    skip_win_index[0], skip_win_index[1],
                    layer, :, :] = 0  # if the path through skip connection is larger, turn off all attention heads.
                max_comb_attention[skip_win_index[0], skip_win_index[1]] = 0
                comb_skip[attn_win_index[0], attn_win_index[1]] = 0
                argsort_index[skip_win_index[0], skip_win_index[1]] = -1
                indices_mix = top_attention_indices[:, :, layer, :, :].sum(
                    -1).sum(-1) + incl_skip_nodes[:, :, layer]
                assert np.all(indices_mix == 1)
                if layer >= self.num_hidden_layers - 2:
                    print('====')
                    print(comb_skip, max_comb_attention)
                    print(top_attention_indices[0, 0])
                    print(incl_skip_nodes)
                    print(top_indices[0, 0])
                expanding_indices.append(argsort_index)
                attention_comb = max_comb_attention + comb_skip
                attention_comb[reverse_qoi_idx[0], reverse_qoi_idx[1]] *= -1
                combs.append(attention_comb)

        # return np.array(combs), np.array(expanding_indices), reverse_qoi
        return np.array(combs), np.array(
            expanding_indices
        ), reverse_qoi, top_attention_indices, incl_skip_nodes, top_indices


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--bert-dir",
        default='/longterm/{}/data/bert/bert_models_tf2/uncased_L-12_H-768_A-12/'
        .format(USERNAME),
        help="Location of the pre-trained BERT model.")
    parser.add_argument(
        "--output-type",
        default='influence',
        help="Select output to evaluate(prob, gradient, influence)")
    parser.add_argument("--mask_indice",
                        type=int,
                        help="index of the mask position")
    parser.add_argument(
        "--threshold",
        default=0,
        type=int,
        help=
        "threshold of how many nodes/indices to take at each layer, in paper we use 1"
    )
    parser.add_argument('--alpha',
                        default=10,
                        type=int,
                        help='resolution for influence')
    parser.add_argument('--cuda',
                        default=1,
                        type=int,
                        help='cuda device number')
    parser.add_argument('--save_every',
                        default=100,
                        type=int,
                        help='how many datapoints to perform per batch')
    parser.add_argument('--examples_path', default=None, help='examples path')
    parser.add_argument("--sentence_type",
                        default='obj_rel_across_anim',
                        help="sentence structure type")
    parser.add_argument("--baseline_type", default='zero', help="baseline type")
    parser.add_argument("--num_example",
                        default=50,
                        type=int,
                        help="number of examples in each sentence form")
    args = parser.parse_args()
    print(args)
    print('creating examples...')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    np.random.seed(10)

    if not args.examples_path:
        tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(
            args.bert_dir, "vocab.txt"),
                                               do_lower_case=True)
        examples = []
        all_features = load_marvin()
        all_features = [f for f in all_features if f[0] == args.sentence_type]
        maxlen = np.max([len(f[2].split()) + 2 for f in all_features])
        print('max length: ', maxlen)
        arr = np.arange(len(all_features))
        np.random.shuffle(arr)
        all_features = [all_features[a] for a in arr]
        sentence_forms = list(set([f[1] for f in all_features]))
        sentence_form_cnt = {sf: 0 for sf in sentence_forms}
        for features in all_features:
            if sentence_form_cnt[features[1]] < args.num_example:
                try:
                    example = Example(
                        features,
                        tokenizer,
                        maxlen,
                        get_baseline=args.output_type == 'influence')
                    examples.append(example)
                    sentence_form_cnt[features[1]] += 1
                    print(features[1], example.tokens)
                except KeyError:
                    print("skipping", features[3], features[4], "bad wins")
        print('finished creating {} examples'.format(len(examples)))
        outpath = "results/{}_examples_{}.pkl".format(args.sentence_type,
                                                      args.num_example)
        write_pickle(examples, outpath)
    else:
        print('loading existing examples')
        with tf.io.gfile.GFile(args.examples_path, 'rb') as f:
            examples = pkl.load(f)
        for e in examples:
            print(e.features[2])
        maxlen = np.max([len(f.tokens) for f in examples])

    influence_extractor = get_influence_extractor(args.bert_dir,
                                                  maxlen,
                                                  use_stop_gradient=True)

    if len(examples) < args.save_every:
        len_examples = args.save_every
    else:
        len_examples = len(examples)
    model_type = args.bert_dir.split('/')[-2]
    indices = np.delete(np.arange(maxlen), args.mask_indice)
    num_batches = int(len_examples / args.save_every)
    for batches in np.arange(num_batches):
        ## the data is sharded only for memory reasons.
        print('=' * 80)
        print('batch', batches)
        example_batch = examples[batches * args.save_every:(batches + 1) *
                                 args.save_every]

        for threshold in [args.threshold]:
            threshold = np.round(threshold, 2)
            print('threshold', threshold)

            greedy_combs, greedy_indices, reverse_qoi = influence_extractor.get_greedy_influence(
                example_batch,
                args.alpha,
                indices,
                threshold=threshold,
                use_attention_output=False,
                baseline=args.baseline_type,
                reverse_qoi=None,
                mask_indice=args.mask_indice)
            # print(greedy_combs.shape)
            # print(greedy_combs[-1, 0, :, :].max(-1))
            # print(greedy_indices[:, 0, :, 0])
            np.save(
                "results/{}_gradp_{}_{}_test_{}_{}_neg_ind_greedy_{}.npy".
                format(args.sentence_type, args.num_example, args.baseline_type,
                       batches, threshold, model_type), reverse_qoi)

            np.save(
                "results/{}_gradp_{}_{}_test_{}_{}_combs_greedy_{}.npy".format(
                    args.sentence_type, args.num_example, args.baseline_type,
                    batches, threshold, model_type), greedy_combs)
            np.save(
                "results/{}_gradp_{}_{}_test_{}_{}_indices_greedy_{}.npy".
                format(args.sentence_type, args.num_example, args.baseline_type,
                       batches, threshold, model_type), greedy_indices)

            fix_top_indice = np.insert(
                greedy_indices, influence_extractor.num_hidden_layers - 1,
                args.mask_indice, 0)

            greedy_combs_attn, greedy_indices_attn, _ = influence_extractor.get_greedy_influence(
                example_batch,
                args.alpha,
                indices,
                threshold=0,
                use_attention_output=True,
                reverse_qoi=reverse_qoi,
                baseline=args.baseline_type,
                fix_top_indice=fix_top_indice,
                mask_indice=args.mask_indice)

            # # print(greedy_indices_attn)
            np.save(
                "results/{}_gradp_{}_{}_test_{}_{}_combs_attn_greedy_{}.npy".
                format(args.sentence_type, args.num_example, args.baseline_type,
                       batches, threshold, model_type), greedy_combs_attn)
            np.save(
                "results/{}_gradp_{}_{}_test_{}_{}_indices_attn_greedy_{}.npy".
                format(args.sentence_type, args.num_example, args.baseline_type,
                       batches, threshold, model_type), greedy_indices_attn)



if __name__ == "__main__":
    main()
