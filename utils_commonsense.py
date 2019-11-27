from __future__ import absolute_import, division, print_function

import json
import logging
import math
import numpy as np
import tqdm
import collections
from io import open
import os
import sys

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)

class InputExample(object):
  """A single multiple choice question."""

  def __init__(
      self,
      example_id,
      question,
      answers,
      label
  ):
    """Construct an instance."""


    self.example_id = example_id
    self.question = question
    self.answers = answers
    self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_json(cls, input_file):
    """Reads a JSON file."""
    with open(input_file, "r", encoding='utf-8') as f:
      return json.load(f)

  @classmethod
  def _read_jsonl(cls, input_file):
    """Reads a JSON Lines file."""
    with open(input_file, "r", encoding='utf-8') as f:
      return [json.loads(ln) for ln in f]


class CommonsenseQAProcessor(DataProcessor):
  """Processor for the CommonsenseQA data set."""

  SPLITS = ['qtoken', 'rand']
  LABELS = ['A', 'B', 'C', 'D', 'E']

  TRAIN_FILE_NAME = 'train_{split}_split.jsonl'
  DEV_FILE_NAME = 'dev_{split}_split.jsonl'
  TEST_FILE_NAME = 'test_{split}_split_no_answers.jsonl'

  def __init__(self, split):
    if split not in self.SPLITS:
      raise ValueError(
        'split must be one of {", ".join(self.SPLITS)}.')

    self.split = split

  def get_train_examples(self, data_dir):
    train_file_name = self.TRAIN_FILE_NAME.format(split=self.split)

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, train_file_name)),
      'train')

  def get_dev_examples(self, data_dir):
    dev_file_name = self.DEV_FILE_NAME.format(split=self.split)

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, dev_file_name)),
      'dev')

  def get_test_examples(self, data_dir):
    test_file_name = self.TEST_FILE_NAME.format(split=self.split)

    return self._create_examples(
      self._read_jsonl(os.path.join(data_dir, test_file_name)),
      'test')

  def get_labels(self):
    return [0, 1, 2, 3, 4]

  def _create_examples(self, lines, set_type):
    examples = []
    for line in lines:
      example_id = line['id']
      question = line['question']['stem']
      answers = np.array([choice['text']
        for choice in sorted(
            line['question']['choices'],
            key=lambda c: c['label'])
      ])
      # the test set has no answer key so use 'A' as a dummy label
      label = self.LABELS.index(line.get('answerKey', 'A'))

      examples.append(
        InputExample(
          example_id=example_id,
          question=question,
          answers=answers,
          label=label))

    return examples

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=0,
                                 mask_padding_with_zero=True):

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        #print("example.question?", example.question)
        for ending_idx, (question, answers) in enumerate(zip(example.question, example.answers)):

            #######changed question->example.question
            tokens_a = tokenizer.tokenize(example.question)

            tokens_b = None
            if example.question.find("_") != -1:
                #this is for cloze question
                tokens_b = tokenizer.tokenize(example.question.replace("_", answers))
            else:
                tokens_b = tokenizer.tokenize(answers)
                #tokens_b = tokenizer.tokenize(example.question + " " + answers)
                # you can add seq token between quesiotn and ending. This does not make too much difference.
                # tokens_b = tokenizer.tokenize(example.question)
                # tokens_b += [sep_token]
                # if sep_token_extra:
                #     tokens_b += [sep_token]
                # tokens_b += tokenizer.tokenize(ending)

            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            #print("tokens a + sep", tokens)
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]


            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids


            print("tokens", tokens)
            print("label", example.label)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            #print("input_ids", input_ids)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            print("input_ids", input_ids)
            print("input_mask", input_mask)
            print("segment_ids", segment_ids)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids))
        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens)))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id = example.example_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features






def example_to_token_ids_segment_ids_label_ids(
    ex_index,
    example,
    max_seq_length,
    tokenizer):
  """Converts an ``InputExample`` to token ids and segment ids."""
  if ex_index < 5:
    logger.info(f"*** Example {ex_index} ***")
    logger.info("example_id: %s" % (example.example_id))

  question_tokens = tokenizer.tokenize(example.question)
  answers_tokens = map(tokenizer.tokenize, example.answers)

  token_ids = []
  segment_ids = []
  for choice_idx, answer_tokens in enumerate(answers_tokens):
    truncated_question_tokens = question_tokens[
      :max((max_seq_length - 3)//2, max_seq_length - (len(answer_tokens) + 3))]
    truncated_answer_tokens = answer_tokens[
      :max((max_seq_length - 3)//2, max_seq_length - (len(question_tokens) + 3))]

    choice_tokens = []
    choice_segment_ids = []
    choice_tokens.append("[CLS]")
    choice_segment_ids.append(0)
    for question_token in truncated_question_tokens:
      choice_tokens.append(question_token)
      choice_segment_ids.append(0)
    choice_tokens.append("[SEP]")
    choice_segment_ids.append(0)
    for answer_token in truncated_answer_tokens:
      choice_tokens.append(answer_token)
      choice_segment_ids.append(1)
    choice_tokens.append("[SEP]")
    choice_segment_ids.append(1)

    choice_token_ids = tokenizer.convert_tokens_to_ids(choice_tokens)

    token_ids.append(choice_token_ids)
    segment_ids.append(choice_segment_ids)

    if ex_index < 5:
      logger.info("choice %s" % choice_idx)
      logger.info("tokens: %s" % " ".join(
        [t for t in choice_tokens]))
      logger.info("token ids: %s" % " ".join(
        [str(x) for x in choice_token_ids]))
      logger.info("segment ids: %s" % " ".join(
        [str(x) for x in choice_segment_ids]))

  label_ids = [example.label]

  if ex_index < 5:
    logger.info("label: %s (id = %d)" % (example.label, label_ids[0]))

  return token_ids, segment_ids, label_ids


def file_based_convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_file
):
  """Convert a set of ``InputExamples`` to a TFRecord file."""

  # encode examples into token_ids and segment_ids
  token_ids_segment_ids_label_ids = [
    example_to_token_ids_segment_ids_label_ids(
      ex_index,
      example,
      max_seq_length,
      tokenizer)
    for ex_index, example in enumerate(examples)
  ]

  # compute the maximum sequence length for any of the inputs
  seq_length = max([
    max([len(choice_token_ids) for choice_token_ids in token_ids])
    for token_ids, _, _ in token_ids_segment_ids_label_ids
  ])

  # encode the inputs into fixed-length vectors
  # writer = tf.python_io.TFRecordWriter(output_file)

  for idx, (token_ids, segment_ids, label_ids) in enumerate(
      token_ids_segment_ids_label_ids
  ):
    if idx % 10000 == 0:
      logger.info("Writing %d of %d" % (
        idx,
        len(token_ids_segment_ids_label_ids)))

    features = collections.OrderedDict()
    for i, (choice_token_ids, choice_segment_ids) in enumerate(
        zip(token_ids, segment_ids)):
      input_ids = np.zeros(max_seq_length)
      input_ids[:len(choice_token_ids)] = np.array(choice_token_ids)

      input_mask = np.zeros(max_seq_length)
      input_mask[:len(choice_token_ids)] = 1

      segment_ids = np.zeros(max_seq_length)
      segment_ids[:len(choice_segment_ids)] = np.array(choice_segment_ids)

      features[f'input_ids{i}'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(input_ids.astype(np.int64))))
      features[f'input_mask{i}'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(input_mask.astype(np.int64))))
      features[f'segment_ids{i}'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(segment_ids.astype(np.int64))))

    features['label_ids'] = tf.train.Feature(
      int64_list=tf.train.Int64List(value=label_ids))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

  return seq_length





def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
    # length or only pop from context
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            logger.info('Attention! you are removing from token_b (swag task is ok). '
                        'If you are training ARC and RACE (you are poping question + options), '
                        'you need to try to use a bigger max seq length!')
            tokens_b.pop()

processors = {
    "commonqa": CommonsenseQAProcessor,
}