from sklearn.utils import murmurhash3_32
from bitarray import bitarray
import math
from random import randint, random, seed, choices
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import functools
import array
import time
from typing import Callable, Dict
import string

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


def hash_function_factory(m: int, sd: any) -> hash_function:
    return lambda x: int(murmurhash3_32(x, sd) % m)


class FusedXorierFilter:
    """
    FusedXorierFilter is a class representing a newly devised structure called the fused XORier filter, a probabilistic
    data structure that is a generalization of the Bloomier filter with spatial coupling, hash caching, and linear
    construction time.

    Parameters:
        elems: a dictionary of keys mapped to their number of occurrences
        c: ratio of table size to the number of keys
        k: number of hash functions
        q: number of bits per slot
    """
    def __init__(self, elems: Dict[any, any], c: float, k: int, q: int, use_cache=False):
        start = time.time()
        self.m = int(c * len(elems))
        self.k = k
        self.q = q
        self.use_cache = use_cache

        S = {elem for elem in elems}
        self.n = len(S)
        self.w = int(4.8 * (self.n**0.58))  # segment length
        self.num_segs = int(self.m / self.w)  # number of segments

        self.table1 = array.array("B", [0 for _ in range(self.m)])
        self.table2 = [0 for _ in range(self.m)]

        # An additional table for keeping track of elements when peeling singletons.
        self.arr = [set() for _ in range(self.m)]

        res = None
        tries = 0
        while res is None:
            print("finding matching")
            hashes = []
            for _ in range(k):
                # Hash into locations within the segments.
                hashes.append(hash_function_factory(self.w, random() * (2 ** 30) + tries))
            # Pick a random starting segment.
            hashes.append(hash_function_factory(self.num_segs - self.k, random() * (2 ** 30) + tries))
            # Get hash code for M.
            hashes.append(hash_function_factory(2 ** q, random() * (2 ** 30) + tries))

            self.hashes = tuple(hashes)

            for t in S:
                hashes = self.hash_all(t, self.hashes)
                for h, hashval in enumerate(hashes[0:len(hashes) - 2], 0):
                    self.arr[(hashes[len(hashes) - 2] + h) * self.w + hashval].add(t)

            res = self.find_match(S)
            tries += 1
            if tries > 10:
                raise Exception("too many tries with k=", k)

        PI, matching = res
        for t in PI:
            v = elems[t]
            hashes = self.hash_all(t, self.hashes)
            neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]
            M = hashes[len(hashes) - 1]
            l = matching[t]
            L = neighborhood[l]

            self.table1[L] = l ^ M ^ functools.reduce(lambda a, b: a ^ b,
                                                      [self.table1[neighborhood[i]] for i in range(self.k)],
                                                      self.table1[L])
            self.table2[L] = v

        self.buildTime = time.time() - start

    def find_match(self, S):
        """
        Finds a matching of S based on the bloomier filter paper.
        :param S: a set of elements to be inserted
        :return: an ordering and matching of S, None if failed
        """
        print("find match", len(S))
        E = set()
        PI = []
        matching = {}
        singletons = self.singletons(S)

        # Add any singletons at the start.
        queue = []
        for singleton in singletons:
            queue.append(singleton)

        # Look for singletons.
        while len(queue) != 0:
            i = queue.pop(len(queue) - 1)
            if len(self.arr[i]) == 1:
                x = list(self.arr[i])[0]
                E.add(x)
                matching[x] = self.tweak(x, singletons)
                singletons.remove(i)

                hashes = self.hash_all(x, self.hashes)
                for h, hashval in enumerate(hashes[0:len(hashes) - 2], 0):
                    self.arr[(hashes[len(hashes) - 2] + h) * self.w + hashval].remove(x)
                    # Find new singletons following peeling of found ones.
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
            # Different from original Bloomier filter -> should not happen now.
            return None

        PI = PIprime
        for t in E:
            PI.append(t)
        return PI, matching | matchingPrime

    def tweak(self, t, singletons):
        """
        Determine the indices of hash functions for singletons.
        :param t: an element in S
        :param singletons: a set of current singletons
        :return: the index of the hash function in neighborhood, None if not found
        """
        hashes = self.hash_all(t, self.hashes)
        neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]

        for i in range(len(neighborhood)):
            if neighborhood[i] in singletons:
                return i
        return None

    def singletons(self, S):
        """
        Finds all singletons for non-peeled elements.
        :param S: a set of elements to be inserted
        :return: a set of all current singletons
        """
        locFreqs = defaultdict(lambda: 0)
        for t in S:
            hashes = self.hash_all(t, self.hashes)
            neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]
            for n in neighborhood:
                locFreqs[n] += 1
        singles = set()
        for k, v in locFreqs.items():
            if v == 1:
                singles.add(k)
        return singles

    def hash_all(self, t, hashes):
        """
        Get hash codes for each element with or without caching (depending on if the use_cache flag is set).
        :param t: an element to be hashed
        :param hashes: the set of all hashes (window selector, location within window, calculating M)
        :return: the hash codes for all hash functions with input t
        """
        if self.use_cache:
            return self.hash_all_cached(t, hashes)
        return [hashes[i](t) for i in range(len(hashes))]

    @functools.cache
    def hash_all_cached(self, t, hashes):
        """
        Caching all hashes for each element that leads to improved speedup (for a reasonable input size).
        :param t: an element to be hashed
        :param hashes: the set of all hashes (window selector, location within window, calculating M)
        :return: the hash codes for all hash functions with input t
        """
        return [hashes[i](t) for i in range(len(hashes))]

    def find_place(self, t):
        """
        Helper for querying the input element in the fused XORier lookup table.
        :param t: an element to be queried
        :return: the hash code for its fingerprint if found, None otherwise
        """
        hashes = self.hash_all(t, self.hashes)
        neighborhood = [((hashes[len(hashes) - 2] + h) * self.w) + hashes[h] for h in range(len(self.hashes) - 2)]
        M = hashes[len(hashes) - 1]

        l = functools.reduce(lambda a, b: a ^ b, [self.table1[neighborhood[i]] for i in range(self.k)], M)
        if l < self.k:
            L = neighborhood[l]
            return L
        return None

    def lookup(self, t):
        """
        Query for an element.
        :param t: an element to be queried
        :return: its XOR value in the fused XORier table if found, None otherwise
        """
        L = self.find_place(t)
        if L is None:
            return None
        return self.table2[L]

    def set_value(self, t, v):
        """
        Sets the value of an input element t to v in the fused XORier table.
        :param t: input element to be set
        :param v: new value
        :return: True if operation successful, False otherwise
        """
        L = self.find_place(t)
        if L is None:
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

fusedXorier = FusedXorierFilter(train_set, 1.23, 3, 8, False)

print('Fused xorier without cache (c = 1.23, k = 3, q = 8)')
print(get_size(fusedXorier) / total_keys_size, get_size(train_set) / total_keys_size)
print(f'build time: {fusedXorier.buildTime}')

fp = 0
for k in test_set:
    if fusedXorier.lookup(k) is not None and k not in train_set:
        fp += 1

print(f'false positive rate: {fp / len(test_set)}')

fusedXorierWithCache = FusedXorierFilter(train_set, 1.23, 3, 8, True)

print('Fused xorier with cache (c = 1.23, k = 3, q = 8)')
print(get_size(fusedXorierWithCache) / total_keys_size, get_size(train_set) / total_keys_size)
print(f'build time: {fusedXorierWithCache.buildTime}')

fp = 0
for k in test_set:
    if fusedXorierWithCache.lookup(k) is not None and k not in train_set:
        fp += 1

print(f'false positive rate: {fp / len(test_set)}')


rand_dict = defaultdict(int)
for _ in range(3 * 10**6):
    rand_dict[''.join(choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=50))] += 1
num_keys = [250000 * i for i in range(1, 12)]
fusedXorierTimes = [math.log(FusedXorierFilter(dict(list(rand_dict.items())[:k]), 1.23, 3, 8).buildTime, 10) for k in num_keys]
plt.plot([i for i in range(0, 5 * 10**6, 10**6)], [i for i in range(5)])
plt.scatter(num_keys, fusedXorierTimes, s=5)
plt.title("Build Time vs. Number of Keys (Fused XORier)")
plt.xlabel("number of keys")
plt.ylabel("log10 build time")
plt.show()
