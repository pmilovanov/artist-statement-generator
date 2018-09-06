#!/usr/bin/python3
import asyncio
import math
import os

import numpy
import numpy as np
import regex as re
from nltk import WordPunctTokenizer
from keras.utils import Sequence
from tqdm import tqdm
from unidecode import unidecode

FLOATX = "float32"


def load_vocab(filename, maxwords=0):
    """
    Load newline-separated words from file to dict mapping them to unique ids.
    :param maxwords: Max number of words to load. Load all by default.
    Returns (list of words, word->id map)
    """
    pad = "·"  # "<#PAD#>"
    vocab = dict()
    words = []
    counter = 1  # start off with 1 so that embedding matrix's first vector is zero and second is for unknown
    words.append(pad)
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if maxwords > 0 and i + 1 > maxwords:
                break
            word = line.strip()
            words.append(word)
            vocab[word] = counter
            counter += 1

    return words, vocab


def load_embeddings(vocab, dim, filename):
    """
    Load a subset of embedding vectors from file corresponding to vocabulary provided.
    Args:
        vocab: string->int map from words to their ids (id corresponds to vector's row in the resulting embedding
             matrix). All ids > 0.
        dim: embedding vector dimension
        filename: file where each line is a word followed by `dim` floats, all space-separated

    Returns:
        MxN = (len(vocab)+1) x dim numpy embedding matrix.
        The +1 for M is because 0th vector is a zero vector for padding.
    """
    em = np.zeros((len(vocab) + 1, dim), dtype="float32")

    with open(filename, "r", encoding="utf-8") as f:
        for linenum, line in enumerate(f):
            line = unidecode(line)
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
    def __init__(self, vocab, vocab_is_lowercase=False):
        """
        Use toseq() method to convert a string to a sequence of token ids
        :param vocab: word->int map
        """
        self.vocab = vocab
        self.vocab_is_lowercase = vocab_is_lowercase
        self.tokenizer = CustomTokenizer(unicode_to_ascii=False)


    def toseq(self, text, notfound=0):
        """
        Converts a string to a sequence of token ids.
        Args:
            text:
            notfound:

        Returns:
            seq, unknown. Seq is a list of integers, word indices in the vocab. Unknown is a list of integers,
            same number of elements as in seq, 1 if the word is not in the vocab and 0 if it is in the vocab. If a word
            is unknown, corresponding value in seq will be 0.
        """
        seq = []
        aux_bits = []
        for word in self.tokenizer.tokenize(text):

            id, aux_unknown, aux_uppercase = 0, 1, 0

            if self.vocab_is_lowercase:
                lower_word = word.lower()
                if lower_word != word:
                    aux_uppercase = 1
                    word = lower_word

            id = self.vocab.get(word, 0)
            if id != 0: aux_unknown = 0

            seq.append(id)
            aux_bits.append([aux_unknown, aux_uppercase])

        return seq, aux_bits


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


def recursively_list_files(path, ignore=None):
    """Recursively list files under a directory, excluding filenames containing strings in the `ignore` list."""
    if ignore is None:
        ignore = ['/.hg', '/.git']
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


# def load_data_async(path, vocab, pad=32, numfiles=0, lowercase=False):
#     loop = asyncio.get_event_loop()
#
#     queue = asyncio.Queue(maxsize=10, loop=loop)
#
#     async def reader(queue):
#         files = recursively_list_files(path)
#         files = tqdm(files, ascii=True, miniters=50)
#         for i, fname in enumerate(files):
#             if numfiles > 0 and (i + 1) > numfiles:
#                 break
#             async with aiofiles.open(fname, "r", encoding="utf-8") as f:
#                 text = await f.read()
#                 await queue.put(text)
#         await queue.put(None)
#
#     X, Xu = [], []
#     t2s = Text2Seq(vocab, vocab_is_lowercase=lowercase)
#
#     async def processor(queue):
#         while True:
#             text = await queue.get()
#             if text is None: break
#             seq, aux = t2s.toseq(text)
#             X.extend(seq)
#             Xu.extend(aux)
#             X.extend([0] * pad)
#             Xu.extend([[0, 0]] * pad)
#
#     loop.run_until_complete(asyncio.gather(reader(queue), processor(queue)))
#
#     X = np.array(X, dtype="int32")
#     Xu = np.array(Xu, dtype="float32")
#     return X, Xu


def load_data(path, vocab, pad=32, numfiles=0, lowercase=False):
    X, Xu = [], []
    t2s = Text2Seq(vocab, vocab_is_lowercase=lowercase)
    files = recursively_list_files(path)
    for i, fname in enumerate(tqdm(files, ascii=True, mininterval=0.5)):
        if numfiles > 0 and (i + 1) > numfiles:
            break  # Process at most `numfiles` files
        with open(fname, "r", encoding="utf-8") as f:
            text = f.read()
            seq, aux = t2s.toseq(text)
            X.extend(seq)
            Xu.extend(aux)
            X.extend([0] * pad)
            Xu.extend([[0, 0]] * pad)

    X = np.array(X, dtype="int32")
    Xu = np.array(Xu, dtype="float32")
    return X, Xu


eltypemap = {
    int:   "int32",
    float: "float32"
    }


def eltype(a):
    t = type(a)
    if t != list:
        return eltypemap.get(t, None)
    if len(a) == 0:
        return None

    return eltype(a[0])


def pad(a, final_length, left=True):
    if type(a) == list:
        dtype = eltype(a)
        if dtype:
            a = np.array(a, dtype=dtype)
        else:
            raise Exception("a should have int32 or float32 elements")
    elif type(a) == numpy.ndarray:
        dtype = a.dtype
    else:
        raise Exception("a should be a list or a numpy.ndarray")

    if final_length <= len(a): return a
    s = list(a.shape)
    s[0] = final_length - len(a)
    z = np.zeros(tuple(s), dtype=dtype)

    if left:
        return np.concatenate([z, a])
    else:
        return np.concatenate([a, z])


def padleft(a, final_length):
    return pad(a, final_length, left=True)


def padright(a, final_length):
    return pad(a, final_length, left=False)


class ShiftByOneSequence(Sequence):
    def __init__(self, data, seqlen, batch_size):
        self.data = data  # just an N-sized array of ints
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.len = len(data) - seqlen * batch_size

    def __getitem__(self, index):
        seq = self.data[index: index + self.seqlen * self.batch_size + 1]
        X = np.reshape(seq[:-1], (self.batch_size, self.seqlen))
        Y = np.expand_dims(seq[[(i + 1) * self.seqlen for i in range(self.batch_size)]], -1)
        return X, Y

    def __len__(self):
        return self.len


class ShiftByOnePermutedSequence(Sequence):
    def __init__(self, data, seqlen, batch_size, permutation_map, dtype="int32"):
        """

        Args:
            data:
            seqlen:
            batch_size:
            permutation_map: `len(data) - seqlen`-sized list of ints
        """
        self.data = data  # N x `dim` list/array
        self.dim = None
        eltype = type(data[0])
        if eltype == list:
            self.dim = [len(data[0])]
        elif eltype == np.ndarray:
            self.dim = list(data[0].shape)

        self.seqlen = seqlen
        self.batch_size = batch_size
        self.len = len(data) - seqlen * batch_size
        self.permutation_map = permutation_map
        self.dtype = dtype

    def __getitem__(self, index):
        shape_x = [self.batch_size, self.seqlen]
        shape_y = [self.batch_size, 1]
        if self.dim:
            shape_x.extend(self.dim)
            shape_y.extend(self.dim)
        X = np.zeros(tuple(shape_x), dtype=self.dtype)
        Y = np.zeros(tuple(shape_y), dtype=self.dtype)

        for i in range(self.batch_size):
            j = index + i * self.seqlen
            if j > len(self.permutation_map):
                print("index", index)
                print("i", i)
                print("j", j)
                print("len pm", len(self.permutation_map))
            mapped_index = self.permutation_map[j]
            X[i, :] = self.data[mapped_index: mapped_index + self.seqlen]
            Y[i, 0] = self.data[mapped_index + self.seqlen]
        return X, Y

    def __len__(self):
        return self.len


class SpecialSequence(Sequence):
    def __init__(self, data_x, data_xu, seqlen, batch_size):
        assert len(data_x) == len(data_xu)
        self.datalen = len(data_x)
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.new_permutation_map()
        self.seqX = ShiftByOnePermutedSequence(data_x, seqlen, batch_size, self.permutation_map)
        self.seqXu = ShiftByOnePermutedSequence(data_xu, seqlen, batch_size, self.permutation_map)

        self.batch_size = batch_size

    def new_permutation_map(self):
        self.permutation_map = np.random.permutation(self.datalen - self.seqlen)

    def __getitem__(self, index):
        X, Y = self.seqX[index]
        Xu, Yu = self.seqXu[index]

        Yfake = np.zeros(Yu.shape, dtype=FLOATX)

        return [X, Xu, Y, Yu], [Yfake]

    def __len__(self):
        return len(self.seqX)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.new_permutation_map()
        self.seqX.permutation_map = self.permutation_map
        self.seqXu.permutation_map = self.permutation_map


class NegativeSamplingPermutedSequence(Sequence):
    """
    Takes a sequence of ints.
    Produces batches of (i.e add a batch axis at the start):
    X = [seq, is_unknown, sample_indices]
    Y = [[1] + [0]*sample_size]

    where
    sample_size: size of sample including one positive example and `sample_size-1` negative examples.
    seq: subsequence of `data` of size `seqlen`
    is_unknown: same size as seq, 0/1 values for whether the i-th word in seq is unknown/known. If 1, then
        corresponding value in seq should be 0
    sample_indices: array of ints of size `sample_size`, indices of words in the sample. First index corresponds to the
        positive word in ground truth, the rest to negative. Corresponding values in `Y` will always be e.g
        [1,0,0,0,0] for `sample_size==5`.
    """

    def __init__(self, data_x, data_xu, seqlen, batch_size, sample_size, vocab_size, permutation_map=None,
                 new_permutation_map_on_epoch_end=True):
        """

        Args:
            data_x:
            data_xu:
            seqlen:
            batch_size:
            sample_size:
            vocab_size: Important: output sample_indices will contain index `vocab_size`, standing for <UNKNOWN>
            permutation_map: `(len(data_x) - seqlen)`-sized list of ints
            new_permutation_map_on_epoch_end:
        """
        self.dataX = data_x
        self.dataXu = data_xu
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.vocab_size = vocab_size
        self.new_permutation_map_on_epoch_end = new_permutation_map_on_epoch_end

        if permutation_map:
            self.permutation_map = permutation_map
        else:
            self.permutation_map = self.gen_permutation_map()

        self.seqX = ShiftByOnePermutedSequence(self.dataX, self.seqlen, self.batch_size, self.permutation_map,
                                               dtype="float32")
        self.seqXu = ShiftByOnePermutedSequence(self.dataXu, self.seqlen, self.batch_size, self.permutation_map,
                                                dtype="int32")

    def gen_permutation_map(self):
        return np.random.permutation(len(self.dataX) - self.seqlen)

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.new_permutation_map_on_epoch_end:
            print("Making new permutation map!")
            self.permutation_map = self.gen_permutation_map()
            self.seqX.permutation_map = self.permutation_map
            self.seqXu.permutation_map = self.permutation_map

    def make_sample_indices2(self, aY, aYu):
        sample_indices = np.zeros((self.batch_size, self.sample_size), dtype="int32")

        for i in range(aY.shape[0]):
            correct_word = aY[i][0]
            if aYu[i][0] == 1:
                correct_word = self.vocab_size  # this artificial index stands for "unknown"

            while True:
                wrong_words = np.random.randint(self.vocab_size + 1, size=self.sample_size - 1)
                if correct_word not in wrong_words:
                    break
            sample_indices[i][0] = correct_word
            sample_indices[i][1:] = wrong_words

        return sample_indices

    def make_sample_indices(self, aY, aYu):
        sample_indices = np.zeros((self.batch_size, self.sample_size, 2), dtype="int32")

        for i in range(aY.shape[0]):
            correct_word = aY[i][0]
            if aYu[i, 0, 0] == 1:
                correct_word = self.vocab_size  # this artificial index stands for "unknown"

            while True:
                wrong_words = np.random.randint(self.vocab_size + 1, size=self.sample_size - 1)
                if correct_word not in wrong_words:
                    break

            row = np.zeros((self.sample_size), dtype="int32")
            row[0] = correct_word
            row[1:] = wrong_words
            batchidxs = np.ones_like(row) * i

            sample_indices[i] = np.concatenate((np.expand_dims(batchidxs, axis=-1), np.expand_dims(row, axis=-1)), -1)

        return sample_indices

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            X = [seq, is_unknown, sample_indices]
            Y = [[1] + [0]*sample_size]
        """
        aX, aY = self.seqX[index]
        aXu, aYu = self.seqXu[index]

        aux_bits_len = aYu.shape[-1]

        raYu = np.reshape(aYu, (aYu.shape[0], aYu.shape[-1]))

        Y = np.zeros((self.batch_size, self.sample_size + aux_bits_len), dtype="int32")
        Y[:, 0] = 1

        Y[:, -aux_bits_len:] = raYu

        sample_indices = self.make_sample_indices(aY, aYu)
        return [[aX, aXu, sample_indices], [Y]]

    def __len__(self):
        return len(self.seqX)


def squish_distribution(scores, alpha):
    """
    Flatten or squish a multinomial distribution.
    Useful to tweak sequence sampling from an RNN.
    Args:
        scores: output of a softmax
        alpha: param to flatten (a < 1) or squish (a > 1)
    """
    s2 = np.power(scores, alpha)
    total = np.sum(s2)
    return s2 / total


def capitalize(s):
    if len(s) == 0:
        return s
    return s[0].upper() + s[1:]


def unknown_word_percentage(Xu):
    s = np.sum(Xu, axis=0)
    return 100.0 * s[0] / len(Xu)