#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""parser.py: User input argument management"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.0.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = ["jarod.duret@alumni.univ-avignon.fr", "jonathan.heno@alumni.univ-avignon.fr"]
__license__ = "MIT"


import argparse

import const


def parse_args(mode: str) -> argparse.Namespace :
    """
    Parses the user arguments.

    Parameters
    ----------
        process: str
            Type of process to fire (either 'train' or 'score')
    
    Returns
    -------
        argparse.Namespace: The parsed user arguments from the command line.
    
    Raises
    ------
        AssertionError: str
            If no running mode is given.
    """
    assert mode in const.RUNNING_MODES, const.UNKNOWN_RUNNING_MODE(mode)

    parser = argparse.ArgumentParser(
        description="Train or scores a neural network on allocine database."
    )

    if mode == const.TRAIN_MODE:
        parser.add_argument(
            "-cfg",
            '--config',
            type=str,
            help="Path to model config file."
        )
        parser.add_argument(
            "-t"
            "--train",
            type=str,
            help="Path to training set."
        )
        parser.add_argument(
            "-d",
            "--dev",
            type=str,
            help="Path to validation set."
        )
    else:
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            help="Path to directory of the trained model."
        )
        parser.add_argument(
            "-t"
            "--test",
            type=argparse.FileType("r"),
            help="Path to test set."
        )
    
    parser.add_argument(
        "out",
        type=str,
        help="Path to output file.",
    )

    return parser.parse_args()
