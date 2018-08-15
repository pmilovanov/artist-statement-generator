from artstat import util


def test_loadvocab():
    words, vocab = util.load_vocab("testdata/test_vocab.txt")
    assert len(words) == 5
    assert len(vocab) == 5
    assert words[0] == "hello"
    assert vocab["hello"] == 0
    assert vocab["HI"] == 4
    assert words[4] == "HI"
