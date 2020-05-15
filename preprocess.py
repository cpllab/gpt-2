#!/usr/bin/env python
# coding: utf-8
"""
Preprocess the given dataset using the BPE encoder from a pretrained model.
"""

import argparse
from collections import Counter
import glob
import json
import os

from tqdm import tqdm

from encoder import get_encoder, DisabledEncoder


SPECIAL_TOKENS = ["<|endoftext|>"]

parser = argparse.ArgumentParser(
        description="Preprocess the given dataset using the BPE encoder from a pretrained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--dataset", metavar="PATH", type=str, required=True)

parser.set_defaults(bpe=True)
parser.add_argument("--no-bpe", dest="bpe", action="store_false")
parser.add_argument("--vocabulary", type=str, metavar="PATH",
                    help="Specify an explicit vocabulary file for non-BPE encoders")

parser.add_argument("-o", "--output", metavar="PATH", type=str, required=True,
                    help="Path to which to save preprocessed dataset.")


def main(args):
    if args.bpe:
        encoder = get_encoder(args.model_dir)
    else:
        with open(args.vocabulary, "r") as f:
            vocab = json.load(f)
        # TODO handle UNKs
        encoder = DisabledEncoder(vocab)

    with open(args.dataset, "r", encoding="utf-8") as f, \
            open(args.output, "w", encoding="utf-8") as outf:
        for line in f:
            outf.write(" ".join(encoder.encode_to_strings(line.strip())) + "\n")


if __name__ == "__main__":
    main(parser.parse_args())
