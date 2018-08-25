import numpy as np
from numpy import array
from numpy.testing import assert_equal

from artstat import util
from artstat.util import CustomTokenizer


def test_load_vocab():
    words, vocab = util.load_vocab("testdata/test_vocab.txt")
    assert len(words) == 6
    assert len(vocab) == 5
    assert words[1] == "hello"
    assert vocab["hello"] == 1
    assert vocab["HI"] == 5
    assert words[5] == "HI"

    words, vocab = util.load_vocab("testdata/test_vocab.txt", 2)
    assert len(vocab) == 2
    assert "HI" not in vocab.keys()


def test_load_embeddings():
    words, vocab = util.load_vocab("testdata/test_vocab.txt")
    em = util.load_embeddings(vocab, 3, "testdata/test_embedding.txt")

    assert_equal(em, array([[0.0, 0.0, 0.0],
                            [4.1, 4.2, 4.3],
                            [5.1, -5.2, 5.3],
                            [-2.1, 2.2, -2.3],
                            [-3.1, 3.2, 3.333],
                            [10.0, 20.0, 30.0]], dtype="float32"))


# path_glove = "/home/pmilovanov/data/glove/glove.840B.300d.txt"

# path_vocab = "/home/pmilovanov/hg/dl/artstat/vocab.txt"

def test_custom_tokenizer():
    tokenizer = CustomTokenizer()

    s = "   Friends, Romans, (,countrymen)@, ain't you \na jolly bunch."
    tokens = tokenizer.tokenize(s)

    assert tokens == ["Friends", ",", "Romans", ",", "(", ",",
                      "countrymen", ")", "@", ",", "ain", "'", "t",
                      "you", "a", "jolly", "bunch", "."]


def test_seqwindows():
    seq = list(range(2, 8))
    X, Y = util.seqwindows(seq, 3, 2)
    Xe = array([[2, 3, 4],
                [4, 5, 6],
                [6, 7, 0]], dtype="int32")
    Ye = array([[3, 4, 5],
                [5, 6, 7],
                [7, 0, 0]], dtype="int32")
    assert_equal(Xe, X)
    assert_equal(Ye, Y)


def test_text2seq():
    words, vocab = util.load_vocab("testdata/test_vocab.txt")
    t2s = util.Text2Seq(vocab)
    text = "    Ahoy hello world hey HI 2 1 \n meow"
    tokens, unknown = t2s.toseq(text)

    assert tokens == [0, 1, 2, 0, 5, 4, 3, 0]
    assert unknown == [1, 0, 0, 1, 0, 0, 0, 1]


def test_shift_by_one_sequence():
    data = np.arange(24)
    seq = util.ShiftByOneSequence(data, 3, 3)

    assert 15 == len(seq)

    assert_equal(seq[14][0],
                 array([[14, 15, 16],
                        [17, 18, 19],
                        [20, 21, 22]]))
    assert_equal(seq[14][1],
                 array([[17], [20], [23]]))
    assert_equal(seq[0][0],
                 array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]]))
    assert_equal(seq[0][1],
                 array([[3], [6], [9]]))


def test_shift_by_one_permuted_sequence():
    data = np.arange(24)
    seq = util.ShiftByOnePermutedSequence(data, 3, 3, range(21))

    assert 15 == len(seq)

    assert_equal(seq[14][0],
                 array([[14, 15, 16],
                        [17, 18, 19],
                        [20, 21, 22]]))

    assert_equal(seq[14][1],
                 array([[17], [20], [23]]))
    assert_equal(seq[0][0],
                 array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]]))
    assert_equal(seq[0][1],
                 array([[3], [6], [9]]))


def test_negative_sampling_permuted_sequence():
    np.random.seed(0)

    X = list(range(100))
    Xu = [0, 0, 1, 0, 0] * 20
    seqlen = 5
    batch_size = 3
    sample_size = 5
    vocab_size = 5000
    permutation_map = list(range(94))

    nss = util.NegativeSamplingPermutedSequence(X, Xu, seqlen, batch_size,
                                                sample_size, vocab_size,
                                                permutation_map=permutation_map)

    assert 85 == len(nss)

    [rX, rXu, rI], [rY] = nss[0]

    assert_equal(rX,
                 array([[0., 1., 2., 3., 4.],
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.]])
                 )
    assert_equal(rXu,
                 array([[0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0.]]))
    assert_equal(
        rI,
        array([[[0, 5], [0, 2732], [0, 2607], [0, 1653], [0, 3264]],
               [[1, 10], [1, 4931], [1, 4859], [1, 1033], [1, 4373]],
               [[2, 15], [2, 3468], [2, 705], [2, 2599], [2, 2135]]], dtype='int32'))

    pass
