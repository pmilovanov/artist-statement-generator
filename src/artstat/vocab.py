#!/usr/bin/python3

import click
import numpy as np
from tqdm import tqdm

# line = unidecode(line)
from artstat.util import CustomTokenizer, recursively_list_files


def add_to_vocab(vocab, word):
    if not word in vocab.keys():
        vocab[word] = 0
    vocab[word] += 1

def sortedvocab(vocab):
    items = vocab.items()
    sorteditems = sorted(items, key=(lambda x: x[1]), reverse=True)
    return sorteditems
    
def printvocab(vocab):
    for word, count in sortedvocab(vocab):
        print("%30d %s" % (count, word))


def writevocab(vocab, outputfile, sortwords, vocabsize):
    items = []
    if sortwords:
        items = sortedvocab(vocab)
    else:
        items = vocab.items()

    covered = 0
    with open(outputfile, "w") as f:
        for i, (word, count) in enumerate(sortedvocab(vocab)):
            if vocabsize > 0 and i > vocabsize:
                break
            f.write(word)
            f.write("\n")
            covered += count

    return covered

@click.command()
@click.option("--normalize_unicode", default=True,
              help="Convert Unicode chars, e.g punctuation, to their " \
                   "closest ASCII counterparts. Def: true")
@click.option("--maxnumfiles", default=0,
              help="Only process at most this many files. " \
              "Set to zero or don't specify to process all files. Def: all files.")
@click.option("--outputfile", default="",
              help="If provided, will write out a list of vocabulary words " \
                   "to this file, one per line.")
@click.option("--vocabsize", default=0,
              help="Max words to output in vocab. Default=0, output all")
@click.option("--output_word_counts_file", default="",
              help="If provided, write a full histogram of word counts to this filename.")
@click.option("--sortwords", default=True,
              help="When writing to file, sort words by frequency descending. Def: true.")
@click.option("--lowercase", default=False,
              help="Generate lowercase words only.")
@click.option("--quiet", default=False, help="Suppress stdout output")
@click.argument("textpath")
def main(normalize_unicode, maxnumfiles, outputfile, textpath, sortwords, quiet, output_word_counts_file, vocabsize,
         lowercase):
    """Loads texts recursively from TEXTPATH and outputs the vocabulary."""

    def echo(*args):
        if not quiet:
            print(*args)
    
    vocab = dict()

    tokenizer = CustomTokenizer(normalize_unicode)

    echo("Processing files in ", textpath)
    echo("")

    files = recursively_list_files(textpath)
    if not quiet:
        files = tqdm(files, ncols=100, ascii=True)

    maxtokens, sumtokens = 0, 0

    wordsperfile = []
    for i, filename in enumerate(files):
        if maxnumfiles > 0 and i+1 > maxnumfiles:
            break
        with open(filename, "r") as f:
            text = f.read()
            if lowercase:
                text = text.lower()
            tokens = tokenizer.tokenize(text)
            if len(tokens) > maxtokens:
                maxtokens = len(tokens)
            sumtokens += len(tokens)
            wordsperfile.append(len(tokens))
            for word in tokens:
                add_to_vocab(vocab, word)

    wpf = np.array(wordsperfile)

    totaloccurrences = np.sum(wpf)
    echo("")
    echo("Total files:", i)
    echo("Unique words:", len(vocab))
    echo("Total words:", totaloccurrences)
    echo("Max words per file:", np.max(wpf))
    echo("Avg words per file:", np.mean(wpf))
    echo("50th percentile:", np.percentile(wpf, 50.0))
    echo("90th percentile:", np.percentile(wpf, 90.0))
    echo("95th percentile:", np.percentile(wpf, 95.0))
    echo("99th percentile:", np.percentile(wpf, 99.0))
    echo("99.9th percentile:", np.percentile(wpf, 99.9))

    if outputfile:
        covered = writevocab(vocab, outputfile, sortwords, vocabsize)
        echo("Wrote vocab to file:", outputfile)
        echo("Vocab size:", vocabsize)
        echo("Ocurrences covered:", covered)
        echo("Coverage: %.2f%%" % (100.0 * covered / totaloccurrences))
    elif output_word_counts_file:
        hist = dict()  # word count histogram
        for word, count in vocab.items():
            if count not in hist.keys():
                hist[count] = 0
            hist[count] += 1
        sortedhist = sorted(hist.items(), key=(lambda x: x[0]))
        with open(output_word_counts_file, "w") as f:
            wordsum = 0
            for k, v in sortedhist:
                wordsum += v
                f.write("%20d : %15d %15d\n" % (k, v, wordsum))
        echo("Wrote word counts to file:", output_word_counts_file)
    else:
        if not quiet:
            printvocab(vocab)


if __name__=="__main__":
    main()
