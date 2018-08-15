#!/usr/bin/python3
import os

import click
import numpy as np
from tqdm import tqdm

# line = unidecode(line)
from artstat.util import CustomTokenizer


def getfiles(path, ignore=['/.hg', '/.git']):
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

def writevocab(vocab, outputfile, sortwords):
    items = []
    if sortwords:
        items = sortedvocab(vocab)
    else:
        items = vocab.items()

    with open(outputfile, "w") as f:
        for word, count in sortedvocab(vocab):
            f.write(word)
            f.write("\n")

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
@click.option("--sortwords", default=True,
              help="When writing to file, sort words by frequency descending. Def: true.")
@click.option("--quiet", default=False, help="Suppress stdout output")
@click.argument("textpath")
def main(normalize_unicode, maxnumfiles, outputfile, textpath, sortwords, quiet):
    """Loads texts recursively from TEXTPATH and outputs the vocabulary."""

    def echo(*args):
        if not quiet:
            print(*args)
    
    vocab = dict()

    tokenizer = CustomTokenizer(normalize_unicode)

    echo("Processing files in ", textpath)
    echo("")

    files = getfiles(textpath)
    if not quiet:
        files = tqdm(files, ncols=100, ascii=True)

    maxtokens, sumtokens = 0, 0

    wordsperfile = []
    for i, filename in enumerate(files):
        if maxnumfiles > 0 and i+1 > maxnumfiles:
            break
        with open(filename, "r") as f:
            text = f.read()
            tokens = tokenizer.tokenize(text)
            if len(tokens) > maxtokens:
                maxtokens = len(tokens)
            sumtokens += len(tokens)
            wordsperfile.append(len(tokens))
            for word in tokens:
                add_to_vocab(vocab, word)

    wpf = np.array(wordsperfile)


    echo("")
    echo("Total files:", i)
    echo("Unique words:", len(vocab))
    echo("Total words:", np.sum(wpf))
    echo("Max words per file:", np.max(wpf))
    echo("Avg words per file:", np.mean(wpf))
    echo("50th percentile:", np.percentile(wpf, 50.0))
    echo("90th percentile:", np.percentile(wpf, 90.0))
    echo("95th percentile:", np.percentile(wpf, 95.0))
    echo("99th percentile:", np.percentile(wpf, 99.0))
    echo("99.9th percentile:", np.percentile(wpf, 99.9))

    if outputfile:
        writevocab(vocab, outputfile, sortwords)
        echo("Wrote vocab to file:", outputfile)
    else:
        if not quiet:
            printvocab(vocab)
                  
    pass

if __name__=="__main__":
    main()
