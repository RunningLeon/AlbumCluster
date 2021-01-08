#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import sys
import json
import glob
from collections import defaultdict
import fnmatch
import shutil


def wprint(*info):
    txt = ' '.join([str(i) for i in info])
    print("\033[0;31m%s\033[0m" % txt)


def glob_file(src_dir, suffix, recursive=False):
    py_version = sys.version_info.major
    if py_version == 2:
        return  glob_file_python2(src_dir, suffix, recursive)
    elif py_version == 3:
        return glob_file_python3(src_dir, suffix, recursive)
    else:
        raise ValueError('Parsed wrong python version =%d, not in [2, 3]'%(py_version))


def glob_file_python2(src_dir, suffix, recursive=False):

    if recursive:
        matches = []
        for root, dirnames, filenames in os.walk(src_dir):
            for filename in fnmatch.filter(filenames, '*' + suffix):
                matches.append(os.path.join(root, filename))
    else:
        matches = [os.path.join(src_dir, filename) for filename in os.listdir(src_dir) if filename.endswith('.' + suffix)]
        matches = [p for p in matches if os.path.isfile(p)]

    return matches


def glob_file_python3(src_dir, suffix='json', recursive=False):

    if recursive:
        matches = glob.glob(os.path.join(src_dir, '**', '*.' + suffix), recursive=True)
    else:
        matches = glob.glob(os.path.join(src_dir, '*.' + suffix))
    return matches


def CHECK_EXIST(path, typ='f'):
    tps = ['f', 'd']
    assert typ in tps, 'typ %s not in %s'%(typ, tps) 
    assert os.path.exists(path), 'File or directory not exists: ' + path
    if typ == 'f':
        assert os.path.isfile(path), 'Input is not %s: %s' % (typ, path)
    else:
        assert os.path.isdir(path), 'Input is not %s: %s' % (typ, path)


def MAKE_EXIST(path, typ='f'):
    tps = ['f', 'd']
    assert typ in tps, 'typ %s not in %s'%(typ, tps)
    dir_to_check = path
    if typ == 'f':
        dir_to_check = os.path.split(path)[0]

    if not os.path.exists(dir_to_check):
        os.makedirs(dir_to_check)