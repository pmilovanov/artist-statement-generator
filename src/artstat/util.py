#!/usr/bin/python3
import math
import os

import numpy as np
import regex as re
from nltk import WordPunctTokenizer
from tqdm import tqdm
from unidecode import unidecode


def load_vocab(filename, maxwords=0):
    """
    Load newline-separated words from file to dict mapping them to unique ids.
    :param maxwords: Max number of words to load. Load all by default.
    Returns (list of words, word->id map)
    """
    vocab = dict()
    words = []
    counter = 1  # start off with 1 so that embedding matrix's first vector is zero and second is for unknown
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if maxwords > 0 and i > maxwords:
                break
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
    :return: MxN = (len(vocab)+1) x dim numpy embedding matrix. The +1 for M is because 0th vector is a zero vector for padding.
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
        """
        :return: seq, unknown. Seq is a list of integers, word indices in the vocab. Unknown is a list of integers, same number of elements as in seq, 1 if the word is not in the vocab and 0 if it is in the vocab. If a word is unknown, corresponding value in seq will be 0.
        """
        seq = []
        unk = []
        for word in self.tokenizer.tokenize(text):
            id = notfound
            isunknown = 1
            if word in self.vocab.keys():
                id = self.vocab[word]
                isunknown = 0
            seq.append(id)
            unk.append(isunknown)
        return seq, unk


def seqwindows(seq, seqlen=256, stride=128, dtype="int32"):
    # nseq = int(math.ceil(len(seq) / stride))

    nseq = int(math.ceil(max(0, len(seq) - seqlen) / stride)) + 1
    X = np.zeros((nseq, seqlen), dtype=dtype)
    Y = np.copy(X)
    seqa = np.array(seq, dtype=dtype)
    for i in range(nseq):
        startX = i * stride
        endX = min(len(seq), startX + seqlen)
        startY = min(len(seq), startX + 1)
        endY = min(len(seq), endX + 1)
        X[i, 0:endX - startX] = seqa[startX:endX]
        Y[i, 0:endY - startY] = seqa[startY:endY]
    return X, Y


def recursively_list_files(path, ignore=['/.hg', '/.git']):
    """Recursively list files under a directory, excluding filenames containing strings in the `ignore` list."""
    results = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            should_append = True
            for ig in ignore:
                if root.find(ig) > -1 or filename.find(ig) > -1:
                    should_append = False
            if should_append:
                results.append(os.path.join(root, filename))
    return results


def load_data_sequences(path, vocab, seqlen, stride, numfiles=0):
    XX, YY, XXu, YYu = [], [], [], []
    t2s = Text2Seq(vocab)
    files = recursively_list_files(path)
    for i, fname in enumerate(tqdm(files, ascii=True)):
        if numfiles > 0 and (i + 1) > numfiles:
            break  # Process at most `numfiles` files
        with open(fname, "r") as f:
            seq, unk = t2s.toseq(f.read())
            Xi, Yi = seqwindows(seq, seqlen, stride)
            Xui, Yui = seqwindows(unk, seqlen, stride, dtype="float32")
            XX.append(Xi)
            YY.append(Yi)
            XXu.append(Xui)
            YYu.append(Yui)
    X = np.concatenate(XX)
    Y = np.concatenate(YY)
    Xu = np.concatenate(XXu)
    Yu = np.concatenate(YYu)
    return X, Y, Xu, Yu
