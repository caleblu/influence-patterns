# """Runs BERT over input data and writes out its attention maps to disk."""
import collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from transformers import BertTokenizer, glue_convert_examples_to_features

import utils
from utils import load_marvin, ninfl, vinfl, write_pickle

NOUN_POS_DIC = {
    'simple_reflexives': [2],
    'obj_rel_within_inanim': [5],
    'simple_agrmt': [2],
    'prep_anim': [2],
    'obj_rel_no_comp_within_anim': [4],
    'reflexives_across': [2],
    'reflexives_across_who': [2],
    'obj_rel_across_anim': [2],
    'obj_rel_across_anim_who': [2],
    'obj_rel_no_comp_across_inanim': [2],
    'prep_inanim': [2],
    'sent_comp': [5],
    'obj_rel_no_comp_within_inanim': [4, 2],
    'subj_rel': [2, 6],
    'subj_rel_who': [2, 6],
    'vp_coord': [2],
    'obj_rel_within_anim': [5, 2],
    'long_vp_coord': [2, 7],
    'reflexive_sent_comp': [5, 2],
    'simple_npi_inanim': [2],
    'obj_rel_across_inanim': [2],
    'obj_rel_no_comp_across_anim': [2],
    'simple_npi_anim': [2],
    'refl_across_gender': [2]
}


def inflect(noun):
    if noun == '[MASK]':
        return noun
    return ninfl[noun]


class Example(object):
    """Represents a single input sequence to be passed into BERT."""

    def __init__(self,
                 features,
                 tokenizer,
                 max_sequence_length,
                 get_baseline=False):
        self.features = features
        if 'only_mask' not in features[0]:
            pre, target, post = features[2].split('***')
            # self.pre_split = pre.split(' ')
            if 'mask' in target.lower():
                target = ['[MASK]']
            else:
                target = tokenizer.tokenize(target)
            tokens = ['[CLS]'] + tokenizer.tokenize(pre)
            self.target_idx = len(tokens)
            self.word_ids = tokenizer.convert_tokens_to_ids(
                [features[3], features[4]])

            self.tokens = tokens + target + tokenizer.tokenize(post) + ['[SEP]']
        else:
            self.tokens = ['[CLS]'
                          ] + ['[MASK]'] * len(features[2].split()) + ['[SEP]']
            self.target_idx = 1
            self.word_ids = tokenizer.convert_tokens_to_ids(
                [features[3], features[4]])
        # print(self.tokens, features[3], features[4], self.target_idx,
        #       self.tokens.index('[MASK]'))
        self.input_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        self.segment_ids = [0] * len(self.tokens)
        self.input_mask = [1] * len(self.tokens)
        try:
            self.subj_idx = NOUN_POS_DIC[features[0]][0]
        except:
            self.subj_idx = 1
        # self.input_mask[self.target_idx] = 0
        while len(self.input_ids) < max_sequence_length:
            self.input_ids.append(0)
            self.input_mask.append(0)
            self.segment_ids.append(0)

        if get_baseline:
            if self.tokens == 'that':
                self.baseline_word = 'that'
            else:
                self.baseline_word = inflect(self.tokens[self.subj_idx])
            self.subj_baseline_id = tokenizer.convert_tokens_to_ids(
                [self.baseline_word])[0]


class SSTExample(object):

    def __init__(self, d, max_seq_length, tokenizer):
        correct_label = d[0]['label'].numpy()[0]
        #         print(correct_label)
        incorrect_label = np.abs(1 - correct_label)
        self.word_ids = np.array([correct_label, incorrect_label])
        self.input_ids = d[1][0]['input_ids'].numpy()[0][:max_seq_length]
        self.tokens = tokenizer.convert_ids_to_tokens(self.input_ids)
        self.segment_ids = d[1][0]['token_type_ids'].numpy()[0][:max_seq_length]
        self.input_mask = d[1][0]['attention_mask'].numpy()[0][:max_seq_length]


def extract_sst_examples_from_df(tds, max_seq_length):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = glue_convert_examples_to_features(tds, tokenizer, 128, 'sst-2')
    zipped_ds = tf.data.Dataset.zip(
        (tds, ds)).filter(lambda x1, x2: tf.reduce_sum(x2[0]['attention_mask'])
                          <= max_seq_length).batch(1)
    examples = []
    for d in tqdm(zipped_ds):
        examples.append(SSTExample(d, max_seq_length, tokenizer))
    return examples


def extract_sst_examples(max_seq_length):
    tds, info = tfds.load("glue/" + 'sst2', split="validation", with_info=True)
    return extract_sst_examples_from_df(tds, max_seq_length)


def examples_in_batches(examples, batch_size):
    for i in range(1 + ((len(examples) - 1) // batch_size)):
        yield examples[i * batch_size:(i + 1) * batch_size]


def extract_examples(sentence_type,
                     num_example,
                     tokenizer,
                     maxlen=None,
                     shuffle=True,
                     seed=1111,
                     output_path_format=None,
                     feature_condition=lambda x: True,
                     example_condition=lambda x: True,
                     all_features=None):
    examples = []
    cnt = 0
    sentence_structures = collections.defaultdict(int)
    if all_features is None:
        all_features = utils.load_marvin()
    if sentence_type is not 'all':
        all_features = [
            f for f in all_features
            if f[0] == sentence_type and feature_condition(f)
        ]
    if maxlen is None:
        maxlen = np.max([len(f[2].split()) + 2 for f in all_features])
    if shuffle:
        arr = np.arange(len(all_features))
        np.random.seed(seed)
        np.random.shuffle(arr)
        all_features = [all_features[a] for a in arr]
    sentence_forms = list(set([f[1] for f in all_features]))
    print(sentence_forms)
    sentence_form_cnt = {sf: 0 for sf in sentence_forms}

    for features in all_features:
        if num_example == 'all':
            example = Example(features, tokenizer, maxlen, get_baseline=True)
            if example_condition(example):
                examples.append(example)
                sentence_form_cnt[features[1]] += 1
            try:
                example = Example(features,
                                  tokenizer,
                                  maxlen,
                                  get_baseline=True)
                if example_condition(example):
                    examples.append(example)
                    sentence_form_cnt[features[1]] += 1
            except KeyError:
                print("skipping", features[3], features[4], "bad wins")
        else:
            if sentence_form_cnt[features[1]] < num_example:
                try:
                    example = Example(features,
                                      tokenizer,
                                      maxlen,
                                      get_baseline=True)
                    if example_condition(example):
                        examples.append(example)
                        sentence_form_cnt[features[1]] += 1
                except KeyError:
                    pass
    print(sentence_form_cnt)
    maxlen = max([len(e.tokens) for e in examples])
    print('finished creating {} examples'.format(len(examples)))
    if output_path_format is not None:
        outpath = output_path_format.format(sentence_type, num_example)
        write_pickle(examples, outpath)

        print('finished writing {} examples'.format(len(examples)), outpath)
        for i in range(5):
            print(examples[i].tokens)
    return examples, maxlen
