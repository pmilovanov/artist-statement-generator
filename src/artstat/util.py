#!/usr/bin/python3
import math

import numpy as np
import regex as re
from nltk import WordPunctTokenizer
from unidecode import unidecode


def load_vocab(filename):
    """
    Load newline-separated words from file to dict mapping them to unique ids.
    Returns (list of words, word->id map)
    """
    vocab = dict()
    words = []
    counter = 1  # start off with 1 so that embedding matrix's first vector is zero
    with open(filename, "r") as f:
        for line in f:
            word = line.strip()
            words.append(word)
            vocab[word] = counter
            counter += 1

    return words, vocab


def load_embeddings(vocab, dim, filename):
    """
    Load a subset of embedding vectors from file corresponding to vocabulary provided.
    :param vocab: string->int map from words to their ids (id corresponds to vector's row in the resulting embedding matrix). All ids > 0.
    :param dim: embedding vector dimension
    :param filename: file where each line is a word followed by `dim` floats, all space-separated
    :return: (len(vocab)+1) x dim numpy embedding matrix. +1 is because 0th vector is a zero vector for "unknown"
    """
    em = np.zeros((len(vocab) + 1, dim), dtype="float32")

    with open(filename, "r") as f:
        for linenum, line in enumerate(f):
            idx = line.find(' ')
            if idx < 0:
                print("malformed line, no space found: line", linenum)
                continue
            word = line[:idx]
            if word not in vocab:
                continue
            i = vocab[word]

            em[i, :] = np.array(line.strip().split()[1:], dtype="float32")

    return em


class CustomTokenizer:
    def __init__(self, unicode_to_ascii=True, punct_one_token_per_char=True):
        self.unicode_to_ascii = unicode_to_ascii
        self.punct_one_token_per_char = punct_one_token_per_char

        self._re_punct = re.compile("(\p{P})")
        self._tokenizer = WordPunctTokenizer()

    def tokenize(self, text):
        if self.unicode_to_ascii:
            text = unidecode(text)
        if self.punct_one_token_per_char:
            text = re.sub(self._re_punct, "\\1 ", text)
        return self._tokenizer.tokenize(text)


class Text2Seq:
    def __init__(self, vocab):
        """
        Use toseq() method to convert a string to a sequence of token ids
        :param vocab: word->int map
        """
        self.vocab = vocab
        self.tokenizer = CustomTokenizer()

    def toseq(self, text, notfound=0):
        seq = []
        for word in self.tokenizer.tokenize(text):
            id = notfound
            if word in self.vocab.keys():
                id = self.vocab[word]
            seq.append(id)
        return seq


def seqwindows(seq, seqlen=256, stride=128):
    nseq = int(math.ceil(len(seq) / stride))
    X = np.zeros((nseq, seqlen), dtype="int32")
    Y = np.copy(X)

    seqa = np.array(seq, dtype="int32")
    for i in range(nseq):
        startX = i * stride
        endX = min(len(seq), startX + seqlen)
        startY = min(len(seq), startX + 1)
        endY = min(len(seq), endX + 1)

        X[i, 0:endX - startX] = seqa[startX:endX]

        Y[i, 0:endY - startY] = seqa[startY:endY]

    return X, Y

# def load_data(vocab, path)
