import collections
import os
import pickle as pkl

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import tensorflow as tf
from tqdm import tqdm

from bert_inf_model import BERTInfModel
from influence_extractor import InfluenceExtractor
from official.nlp.bert import configs, tokenization

example_tokens = ['CLS'] + [
    'the (author) that the (guard) (likes) [MASK]'.split()
] + ['SEP']

greedy_format = "results/{}_gradp_{}_{}_test_{}_0_{}_greedy_{}.npy"


def convert_to_pattern_data(mask_index,
                            num_word,
                            num_layer,
                            num_data_shards=10,
                            comb_greedy_i_raw=None,
                            comb_greedy_attn_i_raw=None,
                            neg_ind_i_raw=None,
                            indices_greedy_i_raw=None,
                            indices_greedy_attn_i_raw=None):
    """
    reformat data for easy plotting. 
    """
    comb_greedy = np.zeros((0, num_word))
    comb_greedy_attn = np.zeros((0, num_word))

    indices_greedy = np.zeros((0, num_layer, num_word))
    indices_greedy_attn = np.zeros((0, num_layer, num_word))

    inf = np.zeros((0, num_word))

    convert_comb = lambda x: np.insert(x[-1].max(-1), mask_index, 0, 1)
    convert_comb_attn = lambda x: np.insert(x[-1], mask_index, 0, 1)

    convert_ig = lambda x: np.insert(
        np.insert(np.swapaxes(x, 0, 1), mask_index, mask_index, -1), num_layer -
        1, mask_index, 1)
    convert_ig_attn = lambda x: np.insert(np.swapaxes(x, 0, 1), mask_index, -2,
                                          -1)
    for i in range(num_data_shards):
        try:
            if comb_greedy_i_raw is None:
                comb_greedy_i_raw = np.load(
                    greedy_format.format(sentence_type, num_example,
                                         baseline_type, i, 'combs', model_type))
                comb_greedy_attn_i_raw = np.load(
                    greedy_format.format(sentence_type, num_example,
                                         baseline_type, i, 'combs_attn',
                                         model_type))
                neg_ind_i_raw = np.load(
                    greedy_format.format(sentence_type, num_example,
                                         baseline_type, i, 'neg_ind',
                                         model_type))
                indices_greedy_i_raw = np.load(
                    greedy_format.format(sentence_type, num_example,
                                         baseline_type, i, 'indices',
                                         model_type))

                indices_greedy_attn_i_raw = np.load(
                    greedy_format.format(sentence_type, num_example,
                                         baseline_type, i, 'indices_attn',
                                         model_type))
            else:
                assert num_data_shards == 1
            infi = np.insert(comb_greedy_i_raw[0].sum(-1), mask_index, 0, 1)
            neg_ind = np.insert(neg_ind_i_raw, mask_index, 0, 1)
            neg_ind = np.where(neg_ind == 1)
            infi[neg_ind[0], neg_ind[1]] *= -1

            inf = np.vstack((inf, infi))
            comb_greedy_attn = np.vstack(
                (comb_greedy_attn, convert_comb_attn(comb_greedy_attn_i_raw)))
            comb_greedy = np.vstack(
                (comb_greedy, convert_comb(comb_greedy_i_raw)))
            #     print(convert_ig(np.load(greedy_format.format(num_example, baseline_type, i, 'indices'))).shape)
            indices_greedy_i = convert_ig(np.squeeze(indices_greedy_i_raw))
            indices_greedy = np.vstack((indices_greedy, indices_greedy_i))
            indices_greedy_attn_i_raw
            indices_greedy_attn = np.vstack(
                (indices_greedy_attn,
                 convert_ig_attn(indices_greedy_attn_i_raw)))
        except FileNotFoundError:
            pass

    neg_ind = np.where(inf < 0)
    comb_greedy_val = comb_greedy
    comb_greedy_val[neg_ind[0], neg_ind[1]] *= -1
    indice_heatmap = np.zeros(
        (indices_greedy.shape[0], num_word, indices_greedy.shape[1], num_word))
    ige = np.swapaxes(indices_greedy, 1, 2)[..., None].astype(np.int32)
    np.put_along_axis(indice_heatmap, ige, np.ones_like(ige), axis=-1)
    indice_heatmap[neg_ind[0], neg_ind[1], :, :] *= -1
    indice_heatmap_attn = np.ones_like(indice_heatmap) * -2
    # print(indice_heatmap_attn.shape)
    ige_attn = np.swapaxes(indices_greedy_attn, 1, 2)[...,
                                                      None].astype(np.int32)
    # print(ige.shape)
    np.put_along_axis(indice_heatmap_attn, ige, ige_attn, axis=-1)
    return inf, comb_greedy_val, comb_greedy_attn, indice_heatmap, indice_heatmap_attn, indices_greedy, indices_greedy_attn


def get_influence_extractor(bert_dir,
                            max_sequence_length,
                            lm_output_type='predictions',
                            load_ckpt_mode='tf1_mlm',
                            use_stop_gradient=False,
                            model_compression=False):
    bert_config = configs.BertConfig.from_json_file(bert_dir +
                                                    'bert_config.json')
    bert_inf_model = BERTInfModel(bert_config,
                                  max_sequence_length,
                                  max_predictions_per_seq=1,
                                  use_stop_gradient=use_stop_gradient,
                                  model_compression=model_compression)

    if load_ckpt_mode == 'tf1_mlm':
        checkpoint = bert_dir + 'bert_model.ckpt'
    # elif load_ckpt_mode == 'tf2_cls':
    #     checkpoint = bert_dir + 'ckpt-3'
    if 'sst' in bert_dir or load_ckpt_mode == 'tf2_cls':
        checkpoint = bert_dir + 'ckpt-3'
        load_ckpt_mode = 'tf2_cls'

    embedding_model, transformer_encoder, decoder_model = bert_inf_model.build_model(
        lm_output_type=lm_output_type,
        load_ckpt_mode=load_ckpt_mode,
        checkpoint=checkpoint,
    )
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(
        bert_dir, "vocab.txt"),
                                           do_lower_case=True)
    influence_extractor = InfluenceExtractor(
        bert_config,
        embedding_model,
        transformer_encoder,
        decoder_model,
        decoder_type=load_ckpt_mode.split('_')[1],
        tokenizer=tokenizer,
        use_stop_gradient=use_stop_gradient,
        model_compression=model_compression)
    return influence_extractor


def compute_lm_outputs(examples, length, bert_dir, batch_size=512):
    influence_extractor = get_influence_extractor(bert_dir, length)
    lm_outputs, _, _, _, _ = influence_extractor.get_lm_prob_batch(
        examples, batch_size)
    return lm_outputs


def load_example_from_name(examples_name):
    with tf.io.gfile.GFile('./results/{}.pkl'.format(examples_name), 'rb') as f:
        examples = pkl.load(f)
    return examples


def load_examples(example_path):
    print('loading existing examples')
    with tf.io.gfile.GFile(example_path, 'rb') as f:
        examples = pkl.load(f)
    maxlen = np.max([len(f.tokens) for f in examples])
    return examples, maxlen


def get_max_len(examples):
    return np.max([len(f.tokens) for f in examples])


def compute_convergence(examples_name,
                        h,
                        l,
                        bert_dir,
                        sentence_type,
                        max_sequence_length,
                        indices=[None],
                        num_example=1000,
                        batch_size=50,
                        shard_size=100,
                        doi='ig',
                        baseline='mask',
                        perturb_xi=False,
                        agg_verb=False,
                        save_data=False,
                        save_data_version=0,
                        seed=10):
    influence_extractor = get_influence_extractor(bert_dir, max_sequence_length)
    examples = load_example_from_name(examples_name)
    print(len(examples))
    # print([e.features[2] for e in examples])
    grads = []
    lmds = []
    np.random.seed(seed)
    if num_example < len(examples):
        examples = [
            examples[i]
            for i in np.random.choice(len(examples), size=num_example)
        ]
    # if perturb_xi:
    #     pass
    for i in tqdm(np.arange(0, 1, 1.0 / shard_size)):
        print(i)
        grad, lmd, _ = influence_extractor.get_e2e_influence(
            examples,
            batch_size,
            indices,
            baseline=baseline,
            start=i,
            end=i + 1.0 / shard_size,
            endpoint=True,
            return_all_resolutions=True,
            return_wrt2_embedding=True,
            ig=doi == 'ig',
            agg_verb=agg_verb)
        # print(grad.shape)
        # print(grad.sum(-1).max())
        grads.append(grad)
        lmds.append(lmd)
    print(np.array(grads).shape, np.array(lmds).shape)
    grads = np.array(grads)
    lmds = np.array(lmds)
    lmds = np.swapaxes(np.swapaxes(lmds, 0, 1), 2,
                       3).reshape(lmds.shape[1], -1, lmds.shape[2], 2)
    grads = np.swapaxes(np.swapaxes(grads, 0, 1), 2,
                        3).reshape(grads.shape[1], -1, grads.shape[2],
                                   grads.shape[-1])

    if save_data:
        np.save(
            'results/metadata/{}_{}_{}_baseline_{}_H_{}_L_{}_agg_{}_V{}.npy'.
            format('grad_array_words', len(grads), doi, baseline, h, l,
                   agg_verb, save_data_version), grads)
        np.save(
            'results/metadata/{}_{}_{}_baseline_{}_H_{}_L_{}_agg_{}_V{}.npy'.
            format('all_lm_differences', len(lmds), doi, baseline, h, l,
                   agg_verb, save_data_version), lmds)
    return grads, lmds
    # grads_array = np.swapaxes(np.squeeze(np.array(grads)), 0,
    #                           1).reshape(len(examples), -1, max_sequence_length)

    # all_lm_differences_array = np.swapaxes(np.array(lmds), 0,
    #                                        1).reshape(len(examples), -1, 2)
    # print(all_lm_differences_array.shape, grads_array.shape)
    # np.save(
    #     'results/metadata/grad_array_words_{}_{}_baseline_{}_H_{}_L_{}_agg_{}.npy'
    #     .format(len(grads_array), doi, baseline, h, l, agg_verb), grads_array)
    # np.save(
    #     'results/metadata/all_lm_differences_{}_{}_baseline_{}_H_{}_L_{}_agg_{}.npy'
    #     .format(len(grads_array), doi, baseline, h, l,
    #             agg_verb), all_lm_differences_array)
    # return grads_array, all_lm_differences_array


def convert_metadata(grads_array_words,
                     all_lm_differences,
                     doi='ig',
                     sample_res=5):
    print(grads_array_words.shape, all_lm_differences.shape)
    if len(grads_array_words.shape) == 3:
        grads_array = np.expand_dims(grads_array_words, 2)
        all_lm_differences = np.expand_dims(all_lm_differences, 2)
    else:
        grads_array = grads_array_words
    lm_differences = all_lm_differences[:,
                                        -1, :, :] - all_lm_differences[:,
                                                                       0, :, :]
    # lm_differences = all_lm_differences[:, -1, :] - all_lm_differences[:, 0, :]
    # grads_array = grads_array_words
    if doi == 'ig':
        grad_res = []
        lm_diff = lm_differences[:, :, 0] - lm_differences[:, :, 1]
        # lm_diff = lm_differences[:, 0] - lm_differences[:, 1]

        for i in np.arange(10, grads_array.shape[1] + 1, sample_res):
            indices = np.linspace(0, grads_array.shape[1] - 1, i, dtype=int)
            grad_res.append(grads_array[:, indices, :].mean(1))
    return lm_diff, grads_array, all_lm_differences, np.array(
        grad_res), grads_array_words


def load_metadata_from_file(res,
                            doi,
                            baseline,
                            h,
                            l,
                            agg_verb,
                            save_data_version=0,
                            sample_res=5,
                            convert_data=True):
    grads_array_words = np.load(
        'results/metadata/grad_array_words_{}_{}_baseline_{}_H_{}_L_{}_agg_{}_V{}.npy'
        .format(res, doi, baseline, h, l, agg_verb,
                save_data_version)).astype(np.float64)
    # lm_differences = np.load('results/metadata/lm_difference_ig_baseline_mask.npy')
    all_lm_differences = np.load(
        'results/metadata/all_lm_differences_{}_{}_baseline_{}_H_{}_L_{}_agg_{}_V{}.npy'
        .format(res, doi, baseline, h, l, agg_verb,
                save_data_version)).astype(np.float64)
    if convert_data:
        return convert_metadata(grads_array_words,
                                all_lm_differences,
                                doi,
                                sample_res=sample_res)
    else:
        return grads_array_words, all_lm_differences


def load_metadata_expand(doi, baseline, h, l):
    all_lm_differences = np.load(
        'results/metadata/all_lm_differences_expand_{}_baseline_{}_H_{}_L_{}.npy'
        .format(doi, baseline, h, l))
    return np.concatenate(
        (all_lm_differences[:, None, 2:], all_lm_differences[:, None, :2]), 1)


def avg_diff_p(x1, x2):
    return np.mean(avg_diff(x1, x2), 1)


def avg_diff(x1, x2):
    if x1.shape != x2.shape:
        x2 = x2[None, :].repeat(len(x1), 0)
    return np.divide(
        np.abs(x1 - x2), np.abs(x2), out=np.zeros_like(x2), where=x2 != 0) * 100


def plot_convergence(res,
                     doi,
                     baselines,
                     hs,
                     ls,
                     agg_verbs,
                     save_data_version,
                     sample_res=5,
                     threshold_index=150):

    fig1 = go.Figure()
    for baseline, h, l, agg_verb in zip(baselines, hs, ls, agg_verbs):
        lm_diff, grads_array, all_lm_differences, grad_res, grads_array_words = load_metadata_from_file(
            res,
            doi,
            baseline,
            h,
            l,
            agg_verb,
            save_data_version=save_data_version,
            sample_res=sample_res)
        fig1.add_trace(
            go.Scatter(x=np.arange(10, grads_array.shape[1] + 1, sample_res),
                       y=avg_diff_p(grad_res[:, :, 0, :].sum(-1),
                                    lm_diff[:, :].sum(-1)),
                       name='H_{}_L{}_agg_verb_{}_baseline_{}'.format(
                           h, l, agg_verb, baseline)))
        fig2 = ff.create_distplot([
            avg_diff(grad_res[threshold_index - 10, :, 0, :].sum(-1),
                     lm_diff[:, :].sum(-1))
        ], [baseline])

        print(
            avg_diff(grad_res[threshold_index - 10, :, 0, :].sum(-1),
                     lm_diff[:, :].sum(-1)).mean())
        print(
            np.sum((avg_diff(grad_res[threshold_index - 10, :, 0, :].sum(-1),
                             lm_diff[:, :].sum(-1)) < 50)))
        avg_diff_res = avg_diff(grad_res[:, :, 0, :].sum(-1),
                                lm_diff[:, :].sum(-1))
        print(avg_diff_res)
        threshold_res = np.argmin(np.abs(avg_diff_res - 50), 0) + 10
        print('number of examples within 50\% of threshold index',
              np.sum(threshold_res < threshold_index))
        fig2.show()
    fig1.update_layout(width=800,
                       height=500,
                       xaxis_title='Resolution/ ',
                       yaxis_title='% Difference: Influence & Qoi',
                       font=dict(size=18))

    fig1.show()


def get_baseline_output_prob(all_lm_differences, examples):
    correct_verb_dict = collections.defaultdict(list)
    for i, e in enumerate(examples):
        correct_verb_dict[e.features[-1]].append(i)

    baseline_correct_list = all_lm_differences[:, 0, 0, 0]
    baseline_incorr_list = all_lm_differences[:, 0, 0, 1]
    baseline_output_dist = baseline_correct_list - baseline_incorr_list

    end_correct_list = all_lm_differences[:, -1, 0, 0]
    end_incorr_list = all_lm_differences[:, -1, 0, 1]
    end_output_dist = end_correct_list - end_incorr_list

    lmd_dist = end_output_dist - baseline_output_dist
    #     print(
    #         'input type | correct verb | correct verb avg logprob | incorrect avg logprob | abs diff'
    #     )
    #     for e in correct_verb_dict:
    #         print('=====')
    #         idx = np.array(correct_verb_dict[e])
    #         print(('baseline', e, np.round(baseline_correct_list[idx].mean(),2),
    #                np.round(baseline_incorr_list[idx].mean(),2),
    #                np.round(baseline_output_dist[idx].mean(),2)))
    #         print(('output', e, np.round(end_correct_list[idx].mean(),2),
    #                np.round(end_incorr_list[idx].mean(),2), np.round(end_output_dist[idx].mean(),2)))
    dist_lst = []
    for e in correct_verb_dict:
        print(e)
        idx = np.array(correct_verb_dict[e])
        dist_lst.append(end_correct_list[idx] - end_incorr_list[idx])
        fig = ff.create_distplot([end_output_dist[idx]], [e],
                                 show_rug=False,
                                 show_curve=False,
                                 bin_size=0.1)
        fig.update_layout(xaxis=dict(range=[-6, 6]))

        fig.show()

        # fig = ff.create_distplot([baseline_output_dist[idx]], [e],
        #                          show_rug=False,
        #                          show_curve=False,
        #                          bin_size=0.01)
        # fig.update_layout(xaxis=dict(range=[-3, 3]))

        # fig.show()

        fig = ff.create_distplot([lmd_dist[idx]], [e],
                                 show_rug=False,
                                 show_curve=False,
                                 bin_size=0.1)
        fig.update_layout(xaxis=dict(range=[-6, 6]))

        fig.show()


#     fig = ff.create_distplot(dist_lst,list(correct_verb_dict.keys()),show_rug=False, show_curve=False, bin_size=0.1)

#     fig = ff.create_distplot([baseline_output_dist, end_output_dist, end_output_dist - baseline_output_dist],['baseline', 'actual', 'qoi'],show_rug=False, show_curve=False, bin_size=0.1)
#     fig.show()
    return baseline_output_dist, end_output_dist


def get_type_dict(examples, inf=None):

    sentence_type_list = set([e.features[1] for e in examples])
    len_tokens = list(set(np.array([len(e.tokens) for e in examples])))
    # print(len_tokens)
    length = len_tokens[0]
    filtered_indices = [
        i for i in range(len(examples)) if len(examples[i].tokens) == length
    ]
    #     print(len(filtered_indices))
    type_dict = {s: [] for s in sentence_type_list}
    for i, e in enumerate(examples):
        if len(e.tokens) == length:
            type_dict[e.features[1]].append(i)

    return type_dict
