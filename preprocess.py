#!/usr/bin/env python
# coding: utf-8
"""
Preprocess the given dataset using the BPE encoder from a pretrained model.
"""

import argparse

from utils import load_encoder


parser = argparse.ArgumentParser(
        description="Preprocess the given dataset using the BPE encoder from a pretrained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--dataset", metavar="PATH", type=str, required=True)

parser.add_argument("-o", "--output", metavar="PATH", type=str, required=True,
                    help="Path to which to save preprocessed dataset.")


def main(args):
    encoder = load_encoder(args.model_dir)

    with open(args.dataset, "r", encoding="utf-8") as f, \
            open(args.output, "w", encoding="utf-8") as outf:
        for line in f:
            outf.write(" ".join(encoder.encode_to_strings(line.strip())) + "\n")


if __name__ == "__main__":
    main(parser.parse_args())
