from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import math
from random import randint, random, seed
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import functools
import array
import time
from joblib import Parallel, delayed
from typing import Callable, Dict

seed("Baker Comes First!")


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


hash_function = Callable[[int | bytes | str], int]


def hash_function_factory(m: int, seed: any) -> hash_function:
    return lambda x: int(murmurhash3_32(x, seed) % m)


# k - num hash funcs
# q - bits per entry
# m - size of table
# n - num keys
# c - size of table / num keys
# w - segment length
# S - support
# D - domain
# R - Range

class FusedXorierFilter:
    """
    FusedXorierFilter is a class representing a probabilistic data structure that is a generalization of the Bloomier
    filter with spatial coupling and hash caching.
    """
    def __init__(self, elems: Dict[any, any], c: float, k: int, q: int):
        start = time.time()
        self.m = int(c * len(elems))
        self.k = k
        self.q = q

        S = {elem for elem in elems}
        self.n = len(S)
        self.w = int(4.8 * (self.n**0.58))  # segment length
        self.num_segs = int(self.m / self.w)  # number of segments

        self.table1 = array.array("B", [0 for _ in range(self.m)])
        self.table2 = [0 for _ in range(self.m)]

        self.arr = [set() for _ in range(self.m)]

        res = None
        tries = 0
        while res is None:
            print("finding matching")
            hashes = []
            for _ in range(k):
                # hash into locations within the segments
                hashes.append(hash_function_factory(self.w, random() * (2 ** 30) + tries))
            # pick a random starting segment
            hashes.append(hash_function_factory(self.num_segs - self.k, random() * (2 ** 30) + tries))
            # get hash code for M
            hashes.append(hash_function_factory(2 ** q, random() * (2 ** 30) + tries))

            self.hashes = tuple(hashes)

            for t in S:
                hashes = self.hashAll(t, self.hashes)
                for h, hashval in enumerate(hashes[0:len(hashes) - 2], 0):
                    self.arr[(hashes[len(hashes) - 2] + h) * self.w + hashval].add(t)

            res = self.findMatch(S)
            tries += 1
            if tries > 10:
                raise Exception("too many tries with k=", k)

        PI, matching = res
        for t in PI:
            v = elems[t]
            hashes = self.hashAll(t, self.hashes)
            neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]
            M = hashes[len(hashes) - 1]
            l = matching[t]
            L = neighborhood[l]

            self.table1[L] = l ^ M ^ functools.reduce(lambda a, b: a ^ b,
                                                      [self.table1[neighborhood[i]] for i in range(self.k)],
                                                      self.table1[L])
            self.table2[L] = v

        self.buildTime = time.time() - start

    def findMatch(self, S):
        print("find match", len(S))
        E = set()
        PI = []
        matching = {}
        singletons = self.singletons(S)

        queue = []
        for singleton in singletons:
            queue.append(singleton)

        while len(queue) != 0:
            i = queue.pop(len(queue) - 1)
            if len(self.arr[i]) == 1:
                x = list(self.arr[i])[0]
                E.add(x)
                matching[x] = self.tweak(x, singletons)
                singletons.remove(i)

                hashes = self.hashAll(x, self.hashes)
                for h, hashval in enumerate(hashes[0:len(hashes) - 2], 0):
                    self.arr[(hashes[len(hashes) - 2] + h) * self.w + hashval].remove(x)
                    # find new singletons following peeling of found ones
                    if len(self.arr[(hashes[len(hashes) - 2] + h) * self.w + hashval]) == 1:
                        queue.append((hashes[len(hashes) - 2] + h) * self.w + hashval)
                        new_singleton = list(self.arr[(hashes[len(hashes) - 2] + h) * self.w + hashval])[0]
                        new_singleton_loc = (hashes[len(hashes) - 2] + h) * self.w + hashval
                        singletons.add(new_singleton_loc)
                        E.add(new_singleton)
                        matching[new_singleton] = self.tweak(new_singleton, singletons)

        if len(E) == 0:
            return None

        PIprime, matchingPrime = [], {}
        H = S.difference(E)
        if len(H) > 0:
            return None

        PI = PIprime
        for t in E:
            PI.append(t)
        return PI, matching | matchingPrime

    def tweak(self, t, singletons):
        hashes = self.hashAll(t, self.hashes)
        neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]

        for i in range(len(neighborhood)):
            if neighborhood[i] in singletons:
                return i
        return None

    def singletons(self, S):
        locFreqs = defaultdict(lambda: 0)
        for t in S:
            hashes = self.hashAll(t, self.hashes)
            neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]
            for n in neighborhood:
                locFreqs[n] += 1
        singles = set()
        for k, v in locFreqs.items():
            if v == 1:
                singles.add(k)
        return singles

    @functools.cache
    def hashAll(self, t, hashes):
        return [hashes[i](t) for i in range(len(hashes))]

    # @functools.cache
    def findPlace(self, t):
        hashes = self.hashAll(t, self.hashes)
        neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]
        M = hashes[len(hashes) - 1]

        l = functools.reduce(lambda a, b: a ^ b, [self.table1[neighborhood[i]] for i in range(self.k)], M)
        if l < self.k:
            L = neighborhood[l]
            return L
        return None

    def lookup(self, t):
        L = self.findPlace(t)
        if L == None:
            return None
        return self.table2[L]

    def set_value(self, t, v):
        L = self.findPlace(t)
        if L == None:
            return False
        self.table2[L] = v
        return True


data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
querylist = data.Query.dropna()

train_set = defaultdict(int)
test_set = defaultdict(int)
partition_size = .8 * len(querylist)

for i, q in enumerate(querylist):
    if i < partition_size:
        train_set[q] += 1
    else:
        test_set[q] += 1

total_keys_size = sum([get_size(key) for key in train_set])

fusedXorierFilter = FusedXorierFilter(train_set, 1.23, 3, 8)

print(get_size(fusedXorierFilter) / total_keys_size, get_size(train_set) / total_keys_size)
print(f'build time: {fusedXorierFilter.buildTime}')

fp = 0
for k in test_set:
    if fusedXorierFilter.lookup(k) is not None and k not in train_set:
        fp += 1

print(f'false positive rate: {fp / len(test_set)}')