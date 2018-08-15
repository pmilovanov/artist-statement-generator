import numpy as np

from artstat import util
from artstat.util import CustomTokenizer


def test_load_vocab():
    words, vocab = util.load_vocab("testdata/test_vocab.txt")
    assert len(words) == 5
    assert len(vocab) == 5
    assert words[0] == "hello"
    assert vocab["hello"] == 0
    assert vocab["HI"] == 4
    assert words[4] == "HI"


def test_load_embeddings():
    words, vocab = util.load_vocab("testdata/test_vocab.txt")
    em = util.load_embeddings(vocab, 3, "testdata/test_embedding.txt")

    expected = np.array([[4.1, 4.2, 4.3],
                         [5.1, -5.2, 5.3],
                         [-2.1, 2.2, -2.3],
                         [-3.1, 3.2, 3.333],
                         [10.0, 20.0, 30.0]], dtype="float32")

    assert np.array_equal(expected, em)


# path_glove = "/home/pmilovanov/data/glove/glove.840B.300d.txt"

# path_vocab = "/home/pmilovanov/hg/dl/artstat/vocab.txt"

def test_custom_tokenizer():
    tokenizer = CustomTokenizer()

    s = "   Friends, Romans, (,countrymen)@, ain't you \na jolly bunch."
    tokens = tokenizer.tokenize(s)

    assert tokens == ["Friends", ",", "Romans", ",", "(", ",",
                      "countrymen", ")", "@", ",", "ain", "'", "t",
                      "you", "a", "jolly", "bunch", "."]
