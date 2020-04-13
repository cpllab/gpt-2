#!/usr/bin/env python
# coding: utf-8
"""
Prepare a non-BPE encoder vocabulary for the given pre-tokenized dataset.
"""

import argparse
from collections import Counter
import glob
import json
import os

from tqdm import tqdm


SPECIAL_TOKENS = ["<|endoftext|>"]

parser = argparse.ArgumentParser(
        description="Prepare an encoder vocabulary for a given pre-tokenized dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", metavar="PATH", type=str, required=True)
parser.add_argument("--encoding", type=str, default="utf-8")

parser.add_argument("-o", "--output", metavar="PATH", type=str, required=True,
                    help="Path to which to save JSON vocabulary.")


def prepare_vocabulary(path, encoding=None):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    vocab = Counter()
    for path in tqdm(paths):
        with open(path, "r", encoding=encoding) as fp:
            text = fp.read().strip().replace("\n", " \n ")
            vocab.update(text.split(" "))

    for special_token in SPECIAL_TOKENS:
        vocab[special_token] = 0

    # Assign IDs in order of decreasing frequency.
    ret_vocab = {}
    for idx, (tok, freq) in enumerate(vocab.most_common()):
        ret_vocab[tok] = idx

    return ret_vocab


def main(args):
    vocab = prepare_vocabulary(args.dataset, encoding=args.encoding)
    with open(args.output, "w") as f:
        json.dump(vocab, f)


if __name__ == "__main__":
    main(parser.parse_args())
