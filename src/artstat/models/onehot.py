import math
import os.path
import sys

import click
import keras
import numpy as np
import tensorflow as tf
from click import option
from keras import Input, Model
from keras.callbacks import LearningRateScheduler
from keras.layers import (Activation, BatchNormalization, Concatenate, CuDNNLSTM, Dense, Dropout, Embedding,
                          Lambda)

from artstat import util
from artstat.util import Text2Seq, capitalize


def sampling_layer_gather_nd(x):
    import tensorflow as tf
    data, sample_indices = x
    return tf.gather_nd(data, tf.cast(sample_indices, tf.int32))


def make_model(*, emb_matrix, vocab, seqlen, sample_size, lstm_size=256, dense_size=300, dense_layers=3, aux_dim=2,
               dropout_rate=0.1, lstm_layers=2):
    input_x = Input((seqlen,), dtype="int32", name="input_x")
    input_aux = Input((seqlen, aux_dim), dtype="float32", name="input_aux")
    input_sample_indices = Input((sample_size, 2), dtype="int32", name="input_sample_indices")

    emb_layer = Embedding(*emb_matrix.shape, input_length=seqlen, trainable=False, weights=[emb_matrix],
                          name="embedding")
    emb_x = emb_layer(input_x)
    concat_x = Concatenate(name="concat_x")([emb_x, input_aux])
    yhat = concat_x

    for i in range(lstm_layers):
        ret_sequences = (i < lstm_layers - 1)
        layerno = i + 1
        yhat = CuDNNLSTM(lstm_size, return_sequences=ret_sequences, name=('lstm%d' % layerno))(yhat)
        # yhat = BatchNormalization()(yhat)
        # yhat = Dropout(dropout_rate)(yhat)

    # yhat = BatchNormalization()(yhat)

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
write_file_path = click.Path(exists=True, dir_okay=False, resolve_path=True, writable=True)
dir_path = click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True)


@click.group()
def main():
    pass


def flags_resources(f):
    f = option("--glove_file", required=True, type=file_path,
               help="File containing pretrained GloVe word embedding data, e.g one of the files from "
                    "http://nlp.stanford.edu/data/glove.6B.zip")(f)
    f = option("--glove_dims", required=True, type=int,
               help="Dimensionality of GloVe word embedding output vectors. Must match that in the provided "
                    "<glove_file>, e.g 300 for glove.6B.300d.txt")(f)
    return f


def flags_shared(f):
    f = option("--seqlen", default=64, help="Length of input sequence of words to predict from.")(f)
    f = option("--vocab_size", default=10000, help="Only use this many words from the top of the vocabulary file.")(f)
    f = option("--vocab_file", required=True, type=file_path,
               help="Vocabulary file to use to map words to integer ids.")(f)
    f = option("--vocab_is_lowercase", default=True,
               help="If true, will convert the word to lowercase before vocabulary lookup. True by default.")(f)
    return f


def flags_hyperparams(f):
    f = option("--lstm_size", type=int, default=256)(f)
    f = option("--lstm_layers", type=int, default=2)(f)
    f = option("--dense_size", type=int, default=256)(f)
    f = option("--dense_layers", type=int, default=5)(f)
    f = option("--dropout_rate", type=float, default=0.1)(f)
    f = option("--learning_rate_initial", type=float, default=0.01)(f)
    f = option("--learning_rate_decay_rate", type=float, default=0.97)(f)
    f = option("--learning_rate_decay_period", type=int, default=10)(f)
    f = option("--batch_size", type=int, default=128)(f)
    f = option("--sample_size", default=10,
               help=("Sample size to be used for negative sampling. Includes the positive example."))(f)
    return f


def info(*args):
    s = " ".join([str(x) for x in args])
    click.echo(click.style(s, fg='red'))


@main.command('train')
@flags_resources
@flags_shared
@flags_hyperparams
@option("--checkpoint_dir", required=True, type=dir_path)
@option("--starting_model_file", type=file_path)
@option("--training_data_dir", required=True, type=dir_path,
        help="Dir containing training data as text files. All files under this dir will be read recursively.")
@option("--training_max_files", default=0)
@option("--num_epochs", default=5, help="Train for this many epochs")
@option("--starting_epoch", default=0)
@option("--epochs_per_dataset", default=32)
def train(vocab_file, vocab_is_lowercase, glove_file, glove_dims, training_data_dir, training_max_files, checkpoint_dir,
          starting_model_file, seqlen, vocab_size, lstm_size, lstm_layers, dense_size, dense_layers, dropout_rate,
          sample_size, learning_rate_initial, learning_rate_decay_rate, learning_rate_decay_period, batch_size,
          num_epochs, starting_epoch, epochs_per_dataset):
    info("Loading vocabulary from ", vocab_file)
    words, vocab = util.load_vocab(vocab_file, vocab_size)
    info("Loaded", len(vocab), "words")
    # print(vocab['test'])

    if starting_model_file:
        info("Loading model from ", starting_model_file)
        model = keras.models.load_model(starting_model_file)
    else:
        info("Loading embedding matrix")
        emb_matrix = util.load_embeddings(vocab, glove_dims, glove_file)
        info("Creating model")
        model = make_model(emb_matrix=emb_matrix, vocab=vocab, seqlen=seqlen, sample_size=sample_size,
                           lstm_size=lstm_size, dense_size=dense_size, dense_layers=dense_layers,
                           lstm_layers=lstm_layers, dropout_rate=dropout_rate)

    info("Loading training data from", training_data_dir)
    X, Xu = util.load_data(training_data_dir, vocab, pad=seqlen, numfiles=training_max_files,
                           lowercase=vocab_is_lowercase)

    info("Unknown words in training data: %.2f%%" % util.unknown_word_percentage(Xu))

    checkpoint_filepath = "weights.lstm%d.batch%d.glove%d.sample%d.vocab%d.%s.hdf5" % (
        lstm_size, batch_size, glove_dims, sample_size, vocab_size, "default")
    checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filepath)
    info("Will write checkpoints to:", checkpoint_filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, verbose=2, save_best_only=True, monitor='loss')

    def decay(epoch):
        return learning_rate_initial * math.pow(learning_rate_decay_rate,
                                                math.floor(epoch / learning_rate_decay_period))

    decay_scheduler = LearningRateScheduler(decay, verbose=1)

    opt = keras.optimizers.Adam(lr=0.0)
    model.compile(opt, loss='categorical_crossentropy', metrics=["accuracy"])

    train_seq = util.NegativeSamplingPermutedSequence(data_x=X, data_xu=Xu, seqlen=seqlen, batch_size=batch_size,
                                                      sample_size=sample_size, vocab_size=len(vocab) + 1)
    steps_per_epoch = int(math.floor(len(X) / (batch_size * epochs_per_dataset)))

    model.fit_generator(train_seq, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        callbacks=[checkpoint, decay_scheduler], initial_epoch=starting_epoch, verbose=1,
                        use_multiprocessing=False, workers=8, max_queue_size=64)


#######################################################################################################################
@main.command('sample')
@flags_shared
@option("--model_file", type=file_path)
@option("--num_words_to_sample", default=5)
@option("--init_text", prompt="Initialization text", type=str)
def sample(vocab_file, vocab_is_lowercase, seqlen, vocab_size, model_file, num_words_to_sample, init_text):
    cols = 80
    info("Loading vocabulary")
    words, vocab = util.load_vocab(vocab_file, vocab_size)

    t2s = Text2Seq(vocab, vocab_is_lowercase=vocab_is_lowercase)
    X, Xu = t2s.toseq(init_text)
    gen = util.padleft(X, seqlen).tolist()
    genu = util.padleft(Xu, seqlen).tolist()

    model_train = keras.models.load_model(model_file)
    model = make_predict_model(model_train)

    info("=" * 100)
    for i, idx in enumerate(gen):
        word = "<UNK>"
        if genu[i][0] < 0.1:
            word = words[idx]
        if genu[i][1] > 0.9:
            word = util.capitalize(word)
        sys.stdout.write(word + " ")
        sys.stdout.flush()

    print()
    info("=" * 100)
    UNK_IDX = len(words)

    punct = ":-;.,!?'\")"
    punct2 = "-/'(\""

    prev_word = words[gen[-1]]
    word = ""

    chars = 0  # chars printed out on this line so far
    tX = np.zeros((1, seqlen), dtype="int32")
    tXu = np.zeros((1, seqlen, 2), dtype="float32")
    results = []

    for j in range(num_words_to_sample):
        tX[0] = np.array(gen[-seqlen:], "int32")
        tXu[0] = np.array(genu[-seqlen:], "float32")
        z = model.predict([tX, tXu])
        scores, aux = z[0][:-2], z[0][-2:]
        idx = UNK_IDX
        while idx == UNK_IDX:
            idx = np.random.choice(range(len(vocab) + 2), p=scores)

        gen.append(idx)
        genu.append([0.0, aux[1]])
        word = words[idx]
        if aux[1] > 0.5: word = capitalize(word)
        results.append(word)

        if cols - chars < len(word) + 1:
            sys.stdout.write("\n")
            chars = 0
        if punct.find(word) < 0 and punct2.find(prev_word) < 0:
            sys.stdout.write(" ")
            chars += 1
        sys.stdout.write(word)
        chars += len(word)
        sys.stdout.flush()

        prev_word = word

    print()


if __name__ == "__main__":
    main()
