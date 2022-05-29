# A set of functions to extend "os" library to single tuple inputs

from os import path
from os import makedirs as mkdir


def untuple(a):
    if len(a) == 1:
        a = a[0]
    return a


def path_join(a, *p):
    a = untuple(a)
    return path.join(a, *p)


def makedirs(a):
    a = untuple(a)
    return mkdir(a)
