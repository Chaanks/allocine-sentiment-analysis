#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""parser.py: User input argument management"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import argparse

import const


def parse_args(mode: str) -> argparse.Namespace:
    """
    Parses the user arguments.

    Parameters
    ----------
    mode (str):
        Type of process to fire (either 'train' or 'score')
    
    Returns
    -------
    (argparse.Namespace): 
        The parsed user arguments from the command line.
    
    Raises
    ------
    (AssertionError): 
        If no running mode is given.
    """
    assert mode in const.RUNNING_MODES, const.UNKNOWN_RUNNING_MODE(mode)

    if mode == const.TRAIN_MODE:
        desc = """
        Trains a neural network on allocine database given a configuration file.
        """
    elif mode == const.SCORE_MODE:
        desc = """
        Scores a trained model on a set of trials.
        """
    elif mode == const.GEN_MODE:
        desc = """
        Generate a set of dataset files from an input source to another with 
        specified format. 
        NOTE: The input files should all have the same extension (either 'xml' 
        or 'json')
        """

    parser = argparse.ArgumentParser(description=desc)

    if mode == const.TRAIN_MODE:
        parser.add_argument(
            "-cfg",
            "--config",
            type=str,
            help="Path to model config file.",
            required=True,
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            help="Type of model to choose",
            required=True,
            choices=const.MODELS,
        )
        parser.add_argument(
            "-t", "--train", type=str, help="Path to training set.", required=True
        )
        parser.add_argument(
            "-d", "--dev", type=str, help="Path to validation set."
        )
    elif mode == const.SCORE_MODE:
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            help="Path to directory of the trained model.",
            required=True,
        )
        parser.add_argument(
            "-t", "--trials", type=str, help="Path to trials.", required=True
        )
    elif mode == const.GEN_MODE:
        parser.add_argument(
            "-d",
            "--data",
            nargs="+",
            metavar="DSFILE",
            type=str,
            help="Path to dataset file(s).",
            required=True,
        )

        parser.add_argument(
            "-if",
            "--in_format",
            type=str,
            help="Source file format.",
            choices=const.IN_FORMATS,
            required=True,
        )
        parser.add_argument(
            "-of",
            "--out_format",
            type=str,
            help="Output file format.",
            choices=const.OUT_FORMATS,
            required=True,
        )

        parser.add_argument(
            "-sw", "--stopwords", type=str, help=f"[{const.ES_FORMAT.upper()}][{const.TRIALS_FORMAT.upper()}][{const.TRAIN_FORMAT.upper()}] Path to stop words file.",
        )
        parser.add_argument(
            "-e", "--extra", type=str, help=f"[{const.ES_FORMAT.upper()}] Path to movie metadata set.",
        )
        parser.add_argument(
            "-ei", "--es_idx", type=str, help=f"[{const.ES_FORMAT.upper()}] Name of the ES database.",
        )
        parser.add_argument(
            "-std",
            "--standardize",
            action="store_true",
            help=f"[{const.ES_FORMAT.upper()}][{const.TRIALS_FORMAT.upper()}][{const.TRAIN_FORMAT.upper()}] Review content standardization.",
        )

    parser.add_argument(
        "out", type=str, help="Path to output directory.",
    )

    ns = parser.parse_args()
    check_args(mode, ns)

    return ns


def check_args(mode: str, ns: argparse.Namespace):
    """
    Check if user input is valid.

    Parameters
    ----------
    mode (str):
        Type of process to fire (either 'train' or 'score')
    ns (argparse.Namespace)
        Arguments parsed from user input
    
    Returns
    -------
    (argparse.Namespace): 
        The parsed user arguments from the command line.
    
    Raises
    ------
    (ValueError): 
        If user input is wrongly informed.
    """
    if mode == const.TRAIN_MODE:
        pass
    elif mode == const.SCORE_MODE:
        pass
    elif mode == const.GEN_MODE:
        if ns.in_format == ns.out_format:
            raise ValueError(const.SAME_IO_FORMAT)
        if ns.extra and ns.out_format != const.ES_FORMAT:
            raise ValueError(const.EXTRA_WITHOUT_ES_OUTPUT)
        if ns.es_idx and ns.out_format != const.ES_FORMAT:
            raise ValueError(const.ES_IDX_WITHOUT_ES_OUTPUT)
        if ns.out_format == "json" and (ns.extra or ns.stopwords or ns.standardize):
            raise ValueError(const.TOO_MANY_OPTIONS_PROVIDED_FOR_JSON_OUTPUT)
        if ns.standardize and ns.out_format == const.JSON_FORMAT:
            raise ValueError(const.NO_STD_FOR_JSON_FORMAT)
