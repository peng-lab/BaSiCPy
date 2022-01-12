"""CLI entry point for PyBaSiC."""

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()


if __name__ == '__main__':
    main()
