#!/usr/bin/python3

import numpy as np


def load_vocab(filename):
    """
    Load newline-separated words from file to dict mapping them to unique ids.
    Returns (list of words, word->id map)
    """
    vocab = dict()
    words = []
    counter = 0
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
    :param vocab: string->int map from words to their ids (id corresponds to vector's row in the resulting embedding matrix)
    :param dim: embedding vector dimension
    :param filename: file where each line is a word followed by `dim` floats, all space-separated
    :return: len(vocab) x dim numpy embedding matrix
    """
    em = np.zeros((len(vocab), dim), dtype="float32")

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
