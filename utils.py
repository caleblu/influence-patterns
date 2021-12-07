from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import Counter
from collections import defaultdict
import time

import json
import pickle
import time
import tensorflow as tf
import numpy as np
import pickle as pkl
import csv
import inflect
import sys
import os
infl_eng = inflect.engine()


def set_up_environment(mem_frac=None, visible_devices=None, min_log_level='3'):
    """
    A helper function to set up a tensorflow environment.

    Args:
        mem_frac: Fraction of memory to limit the gpu to. If set to None,
                  turns on memory growth instead.
        visible_devices: A string containing a comma-separated list of
                         integers designating the gpus to run on.
        min_log_level: One of 0, 1, 2, or 3.
    """
    if visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(min_log_level)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if mem_frac is not None:
                    memory_limit = int(10000 * mem_frac)
                    config = [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit)
                    ]
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu, config)
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(error)

    print(gpus)
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    gpus = tf.config.experimental.list_logical_devices('GPU')
    print(gpus)

## a dictionary for a list of plural/singular verbs for the purpose of having 
# an aggregated group of verbs as qoi, result not included in paper. 
verb_dict = {
    'sing': [
        2573, 3216, 3268, 3504, 3658, 4212, 4832, 5176, 5829, 5975, 6526, 7365,
        7502, 7719, 8289, 8451, 8480, 9631, 10217, 10229, 10854, 11652, 11680,
        12237, 12671, 14523, 14747, 15701, 16412, 17144, 19906, 21145, 22114,
        25126
    ],
    'plur': [
        2147, 2298, 2444, 2448, 2693, 2868, 2991, 2994, 3191, 3198, 3233, 3280,
        3328, 3637, 4133, 4400, 4553, 4682, 4756, 4839, 4982, 5376, 5437, 5993,
        6073, 6682, 6869, 6978, 7180, 7324, 8155, 10436, 11245, 14315
    ]
}

sentence_types = [
    'obj_rel_across_anim', 'subj_rel', 'sent_comp', 'prep_anim',
    'reflexives_across', 'refl_across_gender'
]
num_words = [11, 11, 10, 11, 11, 13]
mask_indices = [7, 7, 6, 7, 8, 10]
example_templates = [{
    'ss': 'the (author) that the (guard) (likes) [MASK] (young) .',
    'sp': 'the (author) that the (guards) (like) [MASK] (young) .',
    'ps': 'the (authors) that the (guard) (likes) [MASK] (young) .',
    'pp': 'the (authors) that the (guards) (like) [MASK] (young) .'
}, {
    'ss': 'the (author) that (likes) the (guard)  [MASK] (young) .',
    'sp': 'the (author) that (likes) the (guards) [MASK] (young) .',
    'ps': 'the (authors) that (like) the (guard) [MASK] (young) .',
    'pp': 'the (authors) that (like) the (guards) [MASK] (young) .'
}, {
    'ss': 'the (author) (said)  the (guard) [MASK] (young) .',
    'sp': 'the (author) (said)  the (guards) [MASK] (young) .',
    'ps': 'the (authors) (said) the (guard)  [MASK] (young) .',
    'pp': 'the (authors) (said) the (guards)  [MASK] (young) .'
}, {
    'ss': 'the (author) (next) (to) the (guard) [MASK] (young) .',
    'sp': 'the (author) (next) (to) the (guards) [MASK] (young) .',
    'ps': 'the (authors) (next) (to) the (guard)  [MASK] (young) .',
    'pp': 'the (authors) (next) (to) the (guards)  [MASK] (young) .'
}, {
    'ss': 'the (author) that the (guard) (likes)  (hurt) [MASK] .',
    'sp': 'the (author) that the (guards) (like)  (hurt) [MASK] .',
    'ps': 'the (authors) that the (guard) (likes)  (hurt) [MASK] .',
    'pp': 'the (authors) that the (guards) (like)  (hurt) [MASK] .'
}, {
    'female_male':
        '(some) (wizard) (who) (can) (dress) (our) (man) (can) (clean) [MASK]',
    'male_male':
        '(some) (king) (who) (can) (dress) (our) (man) (can) (clean) [MASK]',
    'female_female':
        '(some) (wizard) (who) (can) (dress) (our) (woman) (can) (clean) [MASK]',
    'male_female':
        '(some) (king) (who) (can) (dress) (our) (man) (can) (clean) [MASK]'
}]

example_token_templates = [
    'the1 SUBJ that the2 ATTRACTOR VERB [MASK] ADJ .',
    'the1 SUBJ that VERB the2 ATTRACTOR [MASK] ADJ .',
    'the2 ATTRACTOR VERB the1 SUBJ [MASK] ADJ .',
    'the1 SUBJ PREP1 PREP2 the2 ATTRACTOR [MASK] ADJ .',
    'the1 SUBJ that the2 ATTRACTOR VERB1 VERB2 [MASK] .',
    'DEF SUBJ who VERB1 VERB2 DEF ATTRACTOR VERB2 VERB4 [MASK]'
]


def convert_sentence_type(st):
    if 'female' in st or 'male' in st:
        return st
    else:
        st = [s[0] for s in st.split('_') if s == 'sing' or s == 'plur']
        return ''.join(st)


def load_json(path):
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def write_json(o, path):
    tf.io.gfile.makedirs(path.rsplit('/', 1)[0])
    with tf.io.gfile.GFile(path, 'w') as f:
        json.dump(o, f)


def load_pickle(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        return pickle.load(f)


def write_pickle(o, path):
    if '/' in path:
        tf.io.gfile.makedirs(path.rsplit('/', 1)[0])
    with tf.io.gfile.GFile(path, 'wb') as f:
        pickle.dump(o, f, -1)


def logged_loop(iterable, n=None, **kwargs):
    if n is None:
        n = len(iterable)
    ll = LoopLogger(n, **kwargs)
    for i, elem in enumerate(iterable):
        ll.update(i + 1)
        yield elem


def gen_inflect_from_vocab(vocab_file, freq_threshold=1000):
    vbp = {}
    vbz = {}
    nn = {}
    nns = {}
    from_pos = {'NNS': nns, 'NN': nn, 'VBP': vbp, 'VBZ': vbz}

    for line in open(vocab_file):
        if line.startswith(' '):  # empty string token
            continue
        word, pos, count = line.strip().split()
        count = int(count)
        if len(word) > 1 and pos in from_pos and count >= freq_threshold:
            from_pos[pos][word] = count

    verb_infl = {'VBP': 'VBZ', 'VBZ': 'VBP'}
    for word, count in vbz.items():
        candidate = infl_eng.plural_verb(word)
        if candidate in vbp:
            verb_infl[candidate] = word
            verb_infl[word] = candidate

    noun_infl = {'NN': 'NNS', 'NNS': 'NN'}
    for word, count in nn.items():
        candidate = infl_eng.plural_noun(word)
        if candidate in nns:
            noun_infl[candidate] = word
            noun_infl[word] = candidate
    noun_infl['that'] = 'those'
    return verb_infl, noun_infl


vinfl, ninfl = gen_inflect_from_vocab('wiki.vocab')


def get_accuracy(lm_output, nround=3):
    return np.round(
        ((lm_output[:, 0] - lm_output[:, 1]) > 0).sum() / len(lm_output),
        nround)


def rmse(x1, x2):
    return np.mean(np.sqrt((x1 - x2)**2), -1)


def avg_diff_p(x1, x2):
    return np.mean(np.abs(x1 - x2) / np.abs(x2), -1) * 100


def fill_text_blank(tokens, num_word):
    return [
        str(i) + '/' + e
        for i, e in enumerate(tokens + [' '] * (num_word - len(tokens)))
    ]


class LoopLogger(object):
    """Class for printing out progress/ETA for a loop."""

    def __init__(self,
                 max_value=None,
                 step_size=1,
                 n_steps=25,
                 print_time=True):
        self.max_value = max_value
        if n_steps is not None:
            self.step_size = max(1, max_value // n_steps)
        else:
            self.step_size = step_size
        self.print_time = print_time
        self.n = 0
        self.start_time = time.time()

    def step(self, values=None):
        self.update(self.n + 1, values)

    def update(self, i, values=None):
        self.n = i
        if self.n % self.step_size == 0 or self.n == self.max_value:
            if self.max_value is None:
                msg = 'On item ' + str(self.n)
            else:
                msg = '{:}/{:} = {:.1f}%'.format(
                    self.n, self.max_value, 100.0 * self.n / self.max_value)
                if self.print_time:
                    time_elapsed = time.time() - self.start_time
                    time_per_step = time_elapsed / self.n
                    msg += ', ELAPSED: {:.1f}s'.format(time_elapsed)
                    msg += ', ETA: {:.1f}s'.format(
                        (self.max_value - self.n) * time_per_step)
            if values is not None:
                for k, v in values:
                    msg += ' - ' + str(k) + ': ' + (
                        '{:.4f}'.format(v) if isinstance(v, float) else str(v))
            print(msg)


def load_ml_pickle(pkl_file, outfile_path=''):
    with open(pkl_file, 'rb') as f:
        df = pkl.load(f)
    task = pkl_file.split('.')[-2].split('/')[-1]
    print('writing to: ', outfile_path + task + '.tsv')
    with open(outfile_path + task + '.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for typ in df:
            for d in df[typ]:
                tsv_writer.writerow([task, typ, d[0], d[1]])


def load_ml(tsv_file="marvin_linzen_dataset.tsv"):
    cc = Counter()
    out = []
    for line in open(tsv_file):
        case = line.strip().split("\t")
        # print(case)
        cc[case[1]] += 1
        g, ug = case[-2], case[-1]
        if 'taxi driver' in g or 'admire' in g or 'swim' in g:
            continue
        g = g.split()
        ug = ug.split()
        assert (len(g) == len(ug)), (g, ug)
        diffs = [i for i, pair in enumerate(zip(g, ug)) if pair[0] != pair[1]]
        if (len(diffs) != 1):
            #print(diffs)
            #print(g,ug)
            continue
        assert (len(diffs) == 1), diffs
        gv = g[diffs[0]]  # good
        ugv = ug[diffs[0]]  # bad
        g[diffs[0]] = "***mask***"
        g.append(".")
        out.append([case[0], case[1], " ".join(g), gv, ugv])
    return out


def load_marvin():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
    out = load_ml()
    s_features = []
    file_names = ['refl.3', 'refl.4']
    features_who = []
    for f in out:
        if f[0] in [
                'subj_rel', 'obj_rel_across_anim', 'reflexives_across',
                'obj_rel_within_anim'
        ]:
            f_who_string = f[2].split()
            f_who_string[f_who_string.index('that')] = 'who'
            features_who.append(
                [f[0] + '_who', f[1], ' '.join(f_who_string), f[3], f[4]])
    # print(features_who[0])
    for fn in file_names:
        with open('../bert-opensesame/pytorch_pretrained_bert/data_refl/' + fn,
                  'r') as f:
            rgs = [s.split('\t')[0] for s in f.readlines()]
            for rg in rgs:
                if rg.split()[-1] == 'himself':
                    if '3' in fn:
                        f1 = 'male_male'
                    elif '4' in fn:
                        f1 = 'male_female'
                    correct = 'himself'
                    incorrect = 'herself'
                else:
                    if '3' in fn:
                        f1 = 'female_female'
                    elif '4' in fn:
                        f1 = 'female_male'
                    correct = 'herself'
                    incorrect = 'himself'
                slst = rg.split()
                slst[-1] = '***mask***'
                slst.append('.')

                s_features.append([
                    'refl_across_gender', f1, ' '.join(slst), correct, incorrect
                ])
    return out + features_who + s_features + [[
        'only_mask' + str(i), 'ph_ph', '***mask*** ' * i, 'is', 'are'
    ] for i in np.arange(7, 13)]


def eval_marvin():
    o = load_marvin()
    print(len(o), file=sys.stderr)
    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    for i, (case, tp, s, g, b) in enumerate(o):
        ps = get_probs_for_words(s, g, b)
        if ps is None:
            ps = [0, 1]
        gp = ps[0]
        bp = ps[1]
        print(gp > bp, case, tp, g, b, s)
        if i % 100 == 0:
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()