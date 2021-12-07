from tqdm import trange
import numpy as np
import tensorflow as tf


def greedy_path(attentions,
                examples,
                reverse=False,
                aggr_over_head='max',
                add_residual=False,
                normalize=False,
                return_idx=False):

    ## Aggregate attention socres over multiple heads
    if aggr_over_head == 'max':
        attentions = np.max(attentions, axis=2)
    elif aggr_over_head == 'mean':
        attentions = np.mean(attentions, axis=2)
    else:
        attentions = aggr_over_head(attentions)

    if add_residual:
        residual_att = np.eye(attentions.shape[2])[None, None, :]
        aug_att_mat = attentions + residual_att
        if normalize:
            aug_att_mat = aug_att_mat / np.sum(
                aug_att_mat, axis=-1, keepdims=True)
    else:
        aug_att_mat = attentions

    ## (batch, seq_len, attend_in, attend_out)
    N, L, K, K = attentions.shape
    attentions = np.log(attentions)

    all_example_path_scores = []
    if not reverse:
        for n in trange(N):
            per_attention, per_instance = attentions[n], examples[n]
            target_idx = per_instance.target_idx
            sentence = per_instance.tokens
            all_path_scores = []
            for k in range(len(sentence)):
                path = [sentence[k]]
                path_score = 0
                current_idx = k
                for l in range(1, L - 1):
                    i = np.argmax(per_attention[l, current_idx, :])
                    path_score += per_attention[l, current_idx, i]
                    if return_idx:
                        path.append(i)
                    else:
                        path.append(sentence[i])
                    current_idx = i
                if return_idx:
                    path.append(target_idx)
                else:
                    path.append(sentence[target_idx])
                path_score += per_attention[-1, current_idx, target_idx]

                all_path_scores.append([path, path_score])

            all_example_path_scores.append(all_path_scores)
    else:
        for n in trange(N):
            per_attention, per_instance = attentions[n], examples[n]
            target_idx = per_instance.target_idx
            sentence = per_instance.tokens
            all_path_scores = []
            for k in range(len(sentence)):
                path_score = 0
                path = [target_idx]
                current_idx = target_idx
                for l in range(L - 1, -1, -1):
                    if l == 0:
                        i = k
                    else:
                        i = np.argmax(per_attention[l, :, current_idx])
                    path_score += per_attention[l, i, current_idx]
                    current_idx = i
                    if return_idx:
                        path.append(current_idx)
                    else:
                        path.append(sentence[current_idx])

                all_path_scores.append([path[::-1] + [np.exp(path_score)]])

            all_example_path_scores.append(all_path_scores)

    return all_example_path_scores


def beam_search_path(attentions,
                     examples,
                     aggr_over_head='max',
                     excludings=['[CLS]', '[SEP]', None],
                     beam_width=10,
                     return_all_beams=False):

    if aggr_over_head == 'max':
        attentions = np.max(attentions, axis=2)
    elif aggr_over_head == 'mean':
        attentions = np.mean(attentions, axis=2)
    else:
        attentions = aggr_over_head(attentions)

    N, L, K, K = attentions.shape

    all_example_path_scores = []
    for n in trange(N):
        per_attention, per_instance = attentions[n], examples[n]
        target_idx = per_instance.target_idx
        sentence = per_instance.tokens
        all_path_scores = []
        for k in range(len(sentence)):
            if sentence[k] not in excludings:

                ## First layer
                queue = [[k]]
                current_idx = k
                q_score = np.array([0])

                ## Hidden layers
                for l in range(1, L - 1):
                    new_queue = []
                    new_q_score = []
                    for prev_path, q_s in zip(queue, q_score):
                        for new_node in range(K):
                            new_path = list(prev_path) + [new_node]
                            new_queue.append(new_path)
                            new_q_score.append(q_s +
                                               per_attention[l, prev_path[-1],
                                                             new_node])
                    new_q_score = np.array(new_q_score)
                    new_queue = np.asarray(new_queue)
                    new_path_id = np.argsort(new_q_score)[::-1][:beam_width]

                    queue = new_queue[new_path_id]
                    q_score = new_q_score[new_path_id]
                    queue = list(queue)

                ## Last layer
                new_queue = []
                new_q_score = []
                for prev_path, q_s in zip(queue, q_score):
                    new_queue.append(list(prev_path) + [target_idx])
                    new_q_score.append(q_s + per_attention[-1, prev_path[-1],
                                                           target_idx])

                new_q_score = np.array(new_q_score)
                if not return_all_beams:
                    best_path_id = np.argmax(new_q_score)
                    best_path = new_queue[best_path_id]
                    best_q_score = new_q_score[best_path_id]
                    all_path_scores.append([best_path, best_q_score])
                else:
                    best_path_idx = np.argsort(new_q_score)[::-1][:beam_width]
                    new_queue = np.asarray(new_queue)
                    best_paths = new_queue[best_path_idx]
                    best_q_scores = new_q_score[best_path_idx]
                    all_path_scores.append([best_paths, best_q_scores])

        all_example_path_scores.append(all_path_scores)

    return all_example_path_scores


def dp_path(attentions,
            examples,
            aggr_over_head='mean',
            add_residual=True,
            normalize=True):

    if aggr_over_head == 'max':
        attentions = np.max(attentions, axis=2)
    elif aggr_over_head == 'mean':
        attentions = np.mean(attentions, axis=2)
    else:

        attentions = aggr_over_head(attentions)

    if add_residual:
        residual_att = np.eye(attentions.shape[2])[None, None, :]
        aug_att_mat = attentions + residual_att
        if normalize:
            aug_att_mat = aug_att_mat / np.sum(
                aug_att_mat, axis=-1, keepdims=True)
    else:
        aug_att_mat = attentions

    attentions = aug_att_mat

    N, L, K, K = attentions.shape

    def _find_path(att, target):
        log_prob = np.log(att)  # L, K, K
        all_paths = []
        all_paths_score = []
        for s in range(K):
            dp = np.zeros((L + 1, K))

            # initialize with edges between s to nodes at first layer
            dp[1] = log_prob[0, s]
            for l in range(2, dp.shape[0]):
                if l != dp.shape[0] - 1:
                    for k in range(K):
                        # for the k-th node in l-th layer
                        # dp[l, k] = max_{i \in K} dp[l-1][i]+log_prob[l-1][i][k]
                        dp[l, k] = np.max(dp[l - 1] + log_prob[l - 1, :, k])
                else:
                    dp[l,
                       target] = np.max(dp[l - 1] + log_prob[l - 1, :, target])
            all_paths_score.append(dp[-1, target])

            c_node = target
            path = [c_node]
            for l in range(dp.shape[0] - 1, 0, -1):
                for k in range(K):
                    if dp[l,
                          c_node] == dp[l - 1, k] + log_prob[l - 1, k, c_node]:
                        path.append(k)
                c_node = path[-1]
            path = path[::-1]
            all_paths.append(path)

        all_paths = np.asarray(all_paths)
        all_paths_score = np.asarray(all_paths_score)
        all_paths_score = np.exp(all_paths_score)

        return all_paths_score, all_paths

    all_paths = []
    all_scores = []
    for att, e in zip(attentions, examples):
        try:
            target = e.target_idx
        except:
            target = 0
        paths_score, paths = _find_path(att, target)
        all_paths.append(paths)
        all_scores.append(paths_score)

    # (K, N) (K, N, L+1)
    return np.asarray(all_scores), np.asarray(all_paths), np.asarray(attentions)


def attention_rollout(attentions,
                      examples,
                      aggr_over_head='mean',
                      add_residual=True,
                      normalize=True):
    if aggr_over_head == 'max':
        attentions = np.max(attentions, axis=2)
    elif aggr_over_head == 'mean':
        attentions = np.mean(attentions, axis=2)
    else:
        attentions = aggr_over_head(attentions)
    if add_residual:
        residual_att = np.eye(attentions.shape[2])[None, None, :]
        aug_att_mat = attentions + residual_att
        if normalize:
            aug_att_mat = aug_att_mat / np.sum(
                aug_att_mat, axis=-1, keepdims=True)
    else:
        aug_att_mat = attentions
    for i in range(1, aug_att_mat.shape[1]):
        aug_att_mat[:, i] = tf.matmul(tf.constant(aug_att_mat[:, i]),
                                      tf.constant(aug_att_mat[:,
                                                              i - 1])).numpy()
    target_idx = np.array(
        [examples[n].target_idx for n in range(len(examples))])
    scores = aug_att_mat[np.arange(aug_att_mat.shape[0]), :, target_idx]
    attn_rollout = aug_att_mat[:, -1]

    return attn_rollout, scores


def attention_flow(attentions,
                   examples,
                   aggr_over_head='mean',
                   add_residual=True,
                   normalize=True):
    if aggr_over_head == 'max':
        attentions = np.max(attentions, axis=2)
    elif aggr_over_head == 'mean':
        attentions = np.mean(attentions, axis=2)
    else:
        attentions = aggr_over_head(attentions)

    if add_residual:
        residual_att = np.eye(attentions.shape[2])[None, None, :]
        aug_att_mat = attentions + residual_att
        if normalize:
            aug_att_mat = aug_att_mat / np.sum(
                aug_att_mat, axis=-1, keepdims=True)
    else:
        aug_att_mat = attentions

    def _af(att_mat, tokens):
        adj_mat, labels_to_index = agu.get_adjmat(mat=att_mat,
                                                  input_tokens=tokens)
        G = agu.draw_attention_graph(adj_mat,
                                     labels_to_index,
                                     n_layers=att_mat.shape[0],
                                     length=att_mat.shape[-1])

        output_nodes = []
        input_nodes = []
        for key in labels_to_index:
            if 'L6' in key:
                output_nodes.append(key)
            if labels_to_index[key] < att_mat.shape[-1]:
                input_nodes.append(key)

        flow_values = agu.compute_flows(G,
                                        labels_to_index,
                                        input_nodes,
                                        length=att_mat.shape[-1])
        return flow_values

    AF = []
    for i in trange(attentions.shape[0]):
        att = attentions[i]
        e = examples[i]
        AF.append(_af(att, e.tokens))

    return AF
