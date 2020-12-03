#!/usr/bin/env python3
""" defines function that concatenates two arrays """


def cat_arrays(arr1, arr2):
    """ returns new list that is the concatenation of two arrays """
    cat_array = []
    for i in arr1:
        cat_array.append(i)
    for i in arr2:
        cat_array.append(i)
    return cat_array
