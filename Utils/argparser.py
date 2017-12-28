# -*- coding: utf-8 -*-

"""
Adapt from:
1. http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html
2. https://stackoverflow.com/questions/30487767/check-if-argparse-optional-argument-is-set-or-not
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('bool', help='A boolean value', type=bool)
parser.add_argument('--int', help='A int value', type=int)
parser.add_argument('--float', help='A float value', type=float)

args = parser.parse_args()

if args.bool is not None:
    print(args.bool)

if args.int is not None:
    print(args.int)

if args.float is not None:
    print(args.float)
