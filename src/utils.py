import json
import os

from tensorflow.contrib.training import HParams

import encoder




def load_hparams(model_dir):
    """
    Load HParams from the given checkpoint directory.
    """
    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, "r") as f:
        hparams = HParams(**json.load(f))
    return hparams


def load_encoder(model_dir, hparams):
    """
    Load a model ``Encoder`` from the given checkpoint directory.
    """
    if hparams.encoder == "bpe":
        enc = encoder.get_encoder(model_dir)
    elif hparams.encoder == "word":
        vocabulary_path = os.path.join(model_dir, "encoder.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocabulary = json.load(f)
        enc = encoder.DisabledEncoder(vocab)

    return enc
