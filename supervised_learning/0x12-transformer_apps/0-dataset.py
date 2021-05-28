#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """[summary]
    """

    def __init__(self):
        """[summary]
        """
        self.data_train = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True)
        self.data_valid = tfds.load(
            name='ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

    def tokenize_dataset(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        data_pt = []
        data_en = []
        for pt, en in tfds.as_numpy(data):
            data_pt.append(pt.decode('utf-8'))
            data_en.append(en.decode('utf-8'))
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            data_pt,
            target_vocab_size=2**15)
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            data_en,
            target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en
