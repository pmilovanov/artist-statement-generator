import click
from click import option, argument

import tensorflow.keras as K
import os.path

import sys, imp

from artstat import util
import numpy as np

from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (Reshape, Embedding, CuDNNLSTM, BatchNormalization, Dense, Concatenate, Lambda,
                                     Activation, Dropout)
import tensorflow as tf


def make_model(*, emb_matrix, vocab, seqlen, sample_size, lstm_sizes=None, dense_size=300, dense_layers=3, aux_dim=2,
               dropout_rate=0.1):
    if lstm_sizes is None:
        lstm_sizes = [256, 256]

    input_x = Input((seqlen,), dtype="int32", name="input_x")
    input_aux = Input((seqlen, aux_dim), dtype="float32", name="input_aux")
    input_sample_indices = Input((sample_size, 2), dtype="int32", name="input_sample_indices")

    emb_layer = Embedding(*emb_matrix.shape, input_length=seqlen, trainable=False, weights=[emb_matrix],
                          name="embedding")
    emb_x = emb_layer(input_x)
    concat_x = Concatenate(name="concat_x")([emb_x, input_aux])
    yhat = concat_x

    for i, layer_size in enumerate(lstm_sizes):
        ret_sequences = (i < len(lstm_sizes) - 1)
        layerno = i + 1
        yhat = CuDNNLSTM(layer_size, return_sequences=ret_sequences, name=('lstm%d' % layerno))(yhat)
        yhat = BatchNormalization()(yhat)
        yhat = Dropout(dropout_rate)(yhat)

    for layer in range(1, dense_layers + 1):
        yhat = Dense(300, activation="relu", name=("dense%d" % layer))(yhat)
        yhat = BatchNormalization()(yhat)
        yhat = Dropout(dropout_rate)(yhat)

    # These two layers are special: given the model returned by this function,
    # we can make a model for prediction by taking input_x, input_aux as inputs,
    # and constructing the output by putting softmax on top of out_linear
    # and concatenating it with out_aux.
    yhat_aux = Dense(aux_dim, activation="sigmoid", name="out_aux")(yhat)
    yhat = Dense(len(vocab) + 2, activation="linear", name="out_linear")(yhat)
    # len(vocab)+2 is because the zeroth word is for padding
    # and last word is for "unknown"

    # print(input_sample_indices.dtype, input_sample_indices.shape)
    out_train = Lambda(sampling_layer_gather_nd, name="sampling")([yhat, input_sample_indices])
    out_train = Activation('softmax')(out_train)
    out_train = Concatenate(name="concat_out_train")([out_train, yhat_aux])

    model_train = Model([input_x, input_aux, input_sample_indices], [out_train])

    return model_train


def make_predict_model(model_train):
    # Given the model returned by make_model() above
    # we can make a model for prediction by taking input_x, input_aux as inputs,
    # and constructing the output by putting softmax on top of out_linear
    # and concatenating it with out_aux.
    yhat_aux = model_train.get_layer(name="out_aux").output
    yhat = model_train.get_layer(name="out_linear").output

    out_predict = Activation('softmax')(yhat)
    out_predict = Concatenate(name="concat_out_predict")([out_predict, yhat_aux])

    input_x, input_aux, _ = model_train.inputs
    model_predict = Model([input_x, input_aux], [out_predict])

    return model_predict


######################################################################################################################
file_path = click.Path(exists=True, dir_okay=False, resolve_path=True)
dir_path = click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)


def flags_resources(f):
    f = option("--vocab_file", required=True, type=file_path,
               help="Vocabulary file to use to map words to integer ids.")(f)
    f = option("--vocab_is_lowercase", default=True,
               help="If true, will convert the word to lowercase before vocabulary lookup. True by default.")(f)
    f = option("--glove_file", required=True, type=file_path,
               help="File containing pretrained GloVe word embedding data, e.g one of the files from "
                    "http://nlp.stanford.edu/data/glove.6B.zip")(f)
    f = option("--glove_dims", required=True, type=int,
               help="Dimensionality of GloVe word embedding output vectors. Must match that in the provided "
                    "<glove_file>, e.g 300 for glove.6B.300d.txt")(f)
    return f


def flags_hyperparams(f):
    f = option("--seqlen", default=64, help="Length of input sequence of words to predict from.")(f)
    f = option("--vocab_size", default=10000, help="Only use this many words from the top of the vocabulary file.")(f)
    return f


@click.group()
def main():
    pass


@main.command('train')
@flags_resources
@flags_hyperparams
@option("--training_data_dir", required=True, type=dir_path,
        help="Dir containing training data as text files. All files under this dir will be read recursively.")
def train(vocab_file, vocab_is_lowercase, glove_file, glove_dims, training_data_dir, seqlen, vocab_size):
    vocab, words = util.load_vocab(vocab_file, vocab_size)
    emb_matrix = util.load_embeddings(vocab, glove_dims, glove_file)

    X, Xu = util.load_data(training_data_dir, vocab, pad=seqlen, lowercase=vocab_is_lowercase)
    pass


if __name__ == "__main__":
    main()
