from sklearn.utils import murmurhash3_32
import math
from random import randint, random, seed
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import functools
import array
import time
import numpy as np
import random
import string
import concurrent.futures


def hash_function_factory(m, seed):
    return lambda x: int(murmurhash3_32(x, seed) % m)


class XorFilter:
    """
    XorFilter is a class representing an XOR filter, a static data structure with set membership and linear
    construction time.

    Parameters:
        elems: a set of elements to be inserted during construction time
        c: ratio of size of table to number of keys (n/m)
        d: number of hash functions
        l: number of bits stored per slot
    """
    def __init__(self, elems, c, d, l):
        start = time.time()
        S = {elem for elem in elems}
        self.m = int(c * len(S))
        self.l = l
        self.d = d

        # Initialize a pair of tables for 1) the set of keys, and 2) l-bits per slot (the actual XOR table).
        self.table1 = [set() for _ in range(self.m)]
        self.table2 = array.array("I", [0 for _ in range(self.m)])

        # Hash functions for buckets in table 1.
        self.hashes = tuple([hash_function_factory(self.m, sd) for sd in range(self.d)])

        # Fingerprint of an element for table 2.
        self.f = hash_function_factory(2 ** l, self.d)

        print(f"hash functions: {self.d}, bits per slot: {self.l}, table size: {self.m}")

        # Insert the elements into the filter and calculate fingerprints for storage.
        sigma = self.insert(S)
        if sigma is not None:
            print("Insertion successful")
            self.assign(sigma)
        else:
            print("Insertion failed")

        self.build_time = time.time() - start

    def insert(self, S):
        """
        Inserts the elements in S into the initial table.
        :param S: set of elements
        :return: a list of tuples of elements corresponding to the values that hash to their locations
        """
        # Insert all elements in S to table 1.
        for t in S:
            for i in range(self.d):
                self.table1[self.hashes[i](t)].add(t)

        # Add any singletons at the start.
        q = []
        for i in range(self.m):
            if len(self.table1[i]) == 1:
                q.append(i)
        # Continuing peeling and finding new singletons.
        sigma = []
        while len(q) != 0:
            i = q.pop(0)
            if len(self.table1[i]) == 1:
                x = list(self.table1[i])[0]
                sigma.append((x, i))
                for h in self.hashes:
                    try:
                        self.table1[h(x)].remove(x)
                    except KeyError:
                        continue
                    if len(self.table1[h(x)]) == 1:
                        q.append(h(x))
        return sigma if len(sigma) == len(S) else None

    def assign(self, sigma):
        """
        Conducts XOR operations with hashcodes and fingerprint to store elements in the XOR table.
        :param sigma: a list of tuples of elements corresponding to values that hash to locations in the initial table
        :return: nothing
        """
        for x, i in sigma:
            hash_values = [self.hashes[func](x) for func in range(self.d)]
            bit_values = [self.table2[hash_value] for hash_value in hash_values]
            self.table2[i] = self.f(x) ^ np.bitwise_xor.reduce(bit_values)


class BinaryFusedFilter:
    """
    BinaryFusedFilter is a class representing a binary fused filter, a static data structure with set membership, linear
    construction time, and an improved space usage.

    Parameters:
        elems: a set of elements to insert
        m: size of the binary fused table
        d: number of hash functions
        l: number of bits per slot
    """
    def __init__(self, elems, m, d, l):
        start = time.time()
        S = [elem for elem in set(elems)]
        self.n = len(S)
        self.m = int(m * self.n)
        self.l = l
        self.d = d
        self.s = int(2 ** math.floor(math.log(self.n, 3.33) + 2.25))  # segment length

        print(f"hash functions: {self.d}, bits per slot: {self.l}, input set size: {self.n}, segment size: {self.s}")

        # The binary fuse filter itself.
        self.h = [0 for _ in range(self.m)]

        # Fingerprint hash function.
        self.f = hash_function_factory(2 ** l, seed=0)

        # Select random start segments for each element.
        self.start_segs = {}
        for x in S:
            start_seg = random.randint(0, int(self.m / self.s) - self.d)
            self.start_segs[x] = start_seg

        # Construct the filter.
        sigma, w_hashes = self.preprocess(S, 10)
        if len(sigma) == self.n:
            print("Preprocessing successful")
            self.insert(sigma, w_hashes)
            print("Insertion finished")
        else:
            print("Construction failed")

        self.build_time = time.time() - start

    def preprocess(self, S, max_iter) -> (list, list):
        """
        Add all elements to an initial array for preprocessing to the actual binary fused table.
        :param S: a set of elements
        :param max_iter: max number of tries to give; if exceeded, the binary fused filter could not be created
        :return: a list of tuples of elements corresponding to the values that hash to their locations,
                    another list of the generated window hash functions
        """
        p = []
        w_hashes = []
        it = 0
        while it < max_iter:
            print(f"Insertion iter={it}")

            # Select d hash functions for the current iteration.
            rand_seeds = [random.randint(0, 2**31-1) for _ in range(self.d)]
            w_hashes = [(lambda x, i=i, sd=sd: (i * self.s + hash_function_factory(self.s, seed=sd)(x))) for i, sd in
                        zip(range(self.d), rand_seeds)]

            # Sort the items in S by the first hash.
            S.sort(key=lambda x: self.start_segs[x] * self.s + w_hashes[0](x))

            # Array of sets of same size as fuse filter.
            c = [set() for _ in range(self.m)]

            # Populate c with the hash codes for all items in S.
            for x in S:
                for w_hash in w_hashes:
                    c[self.start_segs[x] * self.s + w_hash(x)].add(x)

            # Find singletons.
            q = []
            for loc, hashcode in enumerate(c, 0):
                if len(hashcode) == 1:
                    q.append(loc)

            # Peel newly formed singletons.
            p = []
            while len(q) != 0:
                i = q.pop(len(q) - 1)
                if len(c[i]) == 1:
                    x = list(c[i])[0]
                    p.append((x, i))
                    for w_hash in w_hashes:
                        c[self.start_segs[x] * self.s + w_hash(x)].remove(x)
                        if len(c[self.start_segs[x] * self.s + w_hash(x)]) == 1:
                            q.append(self.start_segs[x] * self.s + w_hash(x))
            print(len(p))

            # Repeat until p contains all items.
            if len(p) == self.n:
                break
            elif it == max_iter - 1:
                return [], w_hashes

            it += 1

        return p, w_hashes

    def insert(self, p, w_hashes):
        """
        Conducts the XOR operations with the hashcodes in consecutive windows and the fingerprint.
        :param p: a list of tuples of elements corresponding to the values that hash to their locations
        :param w_hashes: window hash functions
        :return: nothing
        """
        while len(p) != 0:
            x, i = p.pop(len(p) - 1)
            locs = [self.h[self.start_segs[x] * self.s + w_hashes[idx](x)] for idx in range(self.d)]
            self.h[i] = self.f(x) ^ np.bitwise_xor.reduce(locs)


data = pd.read_csv('user-ct-test-collection-01.txt', sep="\t")
querylist = data.Query.dropna()

m_vals = [1.16 + 0.02 * i for i in range(22)]
c_vals = [1.24 + 0.02 * i for i in range(18)]
build_times_fused = [BinaryFusedFilter(elems=querylist[:1200000], m=m, d=3, l=8).build_time for m in m_vals]
build_times_xor = [XorFilter(elems=querylist[:1200000], c=c, d=3, l=8).build_time for c in c_vals]
plt.scatter(m_vals, build_times_fused, s=10)
plt.scatter(c_vals, build_times_xor, s=10)
plt.title("Build Time vs. Filter Size for XOR and Binary Fused (AOL Dataset)")
plt.xlabel("Filter Size / Dataset Size")
plt.ylabel("Build Time (s)")
plt.ylim(bottom=0)
plt.legend(labels=["binary fused filter", "xor filter"], loc="upper left")
plt.show()

# Plot build time vs. number of keys
# rand_list = []
# for _ in range(10**7):
#     rand_list.append(''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=50)))
# keys = [1000000 + 100000 * i for i in range(22)]
# build_times_fused = [BinaryFusedFilter(elems=rand_list[:k], m=1.15, d=3, l=8).build_time for k in keys]
# build_times_xor = [XorFilter(elems=rand_list[:k], c=1.23, d=3, l=8).build_time for k in keys]
# plt.scatter(keys, build_times_fused, s=5)
# plt.scatter(keys, build_times_xor, s=5)
# plt.title("Build Time vs. Number of Keys")
# plt.xlabel("Number of keys")
# plt.ylabel("Build time")
# plt.legend(labels=["binary fused filter", "xor filter"], loc="upper left")
# plt.show()


def build_filter(filter_type, elements, **kwargs):
    if filter_type == 'binary fused':
        return BinaryFusedFilter(elems=elements, **kwargs).build_time
    elif filter_type == 'xor':
        return XorFilter(elems=elements, **kwargs).build_time
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def parallel_builds(filter_type, elements, num_keys, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(build_filter, filter_type, elements[:k], **kwargs): k for k in num_keys}
        concurrent.futures.wait(futures)
        build_times = {k: future.result() for future, k in futures.items()}
    return build_times


rand_list = []
for _ in range(10**7):
    rand_list.append(''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=50)))
num_keys = [1000000 + 200000 * i for i in range(5)]
binary_fused_kwargs = {'m': 1.14, 'd': 3, 'l': 8}
xor_kwargs = {'c': 1.23, 'd': 3, 'l': 8}
build_times_fused = parallel_builds('binary fused', rand_list, num_keys, **binary_fused_kwargs)
build_times_xor = parallel_builds('xor', rand_list, num_keys, **xor_kwargs)
plt.scatter(num_keys, list(build_times_fused.values()), s=5)
plt.scatter(num_keys, list(build_times_xor.values()), s=5)
plt.title("Build Time vs. Number of Keys")
plt.xlabel("Number of keys")
plt.ylabel("Build time")
plt.legend(labels=["binary fused filter", "xor filter"], loc="upper left")
plt.show()





# TEMP CODE

# class CoupledXorFilter:
#     def __init__(self, elems, c, d, l, w):
#         start = time.time()
#         S = {elem for elem in elems}
#         self.m = int(c * len(S))
#         self.l = l
#         self.d = d
#         self.w = int(w * self.m)
#         if self.m % self.w != 0:
#             while self.m % self.w != 0:
#                 self.m -= 1
#
#         # mapping of keys to hash values
#         self.w_table = defaultdict(list)
#
#         # table for storing sets of keys
#         self.table1 = [set() for _ in range(self.m)]
#
#         # final table for xor operations
#         self.table2 = array.array("I", [0 for _ in range(self.m)])
#
#         # array of slots to choose from
#         self.slots = [i * self.w for i in range(math.ceil(self.m / self.w))]
#
#         # hash function for choosing a window at random
#         self.h_w = hash_function_factory(len(self.slots), 0)
#         # hash functions for hashing into slots within a window
#         self.w_hashes = tuple([hash_function_factory(self.w, sd) for sd in range(self.d + 1, 2 * self.d + 1)])
#         # hash functions for truncated window
#         self.w_trunc_hashes = tuple([hash_function_factory(self.m - (len(self.slots) - 1) * self.w + 1, sd)
#                                      for sd in range(self.d + 1, 2 * self.d + 1)])
#         # hash functions for hashing into slots in the final xor table
#         self.hashes = tuple([hash_function_factory(self.m, sd) for sd in range(1, self.d + 1)])
#         # hash function for assigning a fingerprint to each key before any xor operations
#         self.f = hash_function_factory(2 ** l, 2 * self.d + 1)
#
#         print("Windows: " + str(self.w) + ", Total Slots: " + str(self.m) + ", Window Slots: " + str(self.slots))
#         print(self.m)
#         print(self.m - (len(self.slots) - 1) * self.w + 1)
#
#         sigma = self.insert(S)
#         if sigma is not None:
#             print("Insertion successful")
#             self.assign(sigma)
#         else:
#             print("Insertion failed")
#
#         self.build_time = time.time() - start
#
#     def insert(self, S):
#         # Insert all elements in S to random windows in table 1.
#         for t in S:
#             hash_values = self.hash_all(t, self.w_hashes)
#             for i in range(len(self.w_hashes)):
#                 w_loc = self.slots[self.h_w(randint(0, 1000))]
#                 self.w_table[t].append(w_loc)
#                 # if self.slots[len(self.slots) - 1] == w_loc:
#                 #     self.table1[w_loc + self.w_trunc_hashes[i](t)].add(t)
#                 # else:
#                 #     self.table1[w_loc + hash_values[i]].add(t)
#                 self.table1[w_loc + hash_values[i]].add(t)
#
#         q = []
#         for i in range(self.m):
#             if len(self.table1[i]) == 1:
#                 q.append(i)
#         sigma = []
#         while len(q) != 0:
#             # print(len(sigma))
#             i = q.pop(0)
#             if len(self.table1[i]) == 1:
#                 x = list(self.table1[i])[0]
#                 sigma.append((x, i))
#                 for idx in range(len(self.w_hashes)):
#                     h = self.w_hashes[idx]
#                     # h = self.w_hashes[idx] if self.w_table[x][idx] != self.slots[len(self.slots) - 1] else \
#                     #         self.w_trunc_hashes[idx]
#                     try:
#                         self.table1[self.w_table[x][idx] + h(x)].remove(x)
#                     except KeyError:
#                         continue
#                     if len(self.table1[self.w_table[x][idx] + h(x)]) == 1:
#                         q.append(self.w_table[x][idx] + h(x))
#
#         # print(len(S))
#         # # Insert all elements in S to random windows in table 1.
#         # for t in S:
#         #     hash_values = self.hash_all(t, self.w_hashes)
#         #     w_loc = self.slots[self.h_w(randint(0, 1000))]
#         #     for i in range(len(self.w_hashes)):
#         #         self.table1[w_loc + hash_values[i]].add(t)
#         #
#         # sigma = []
#         # q = []
#         # while len(sigma) < len(S):
#         #     print(len(sigma))
#         #     w_loc = self.slots[self.h_w(randint(0, 1000))]
#         #     print(w_loc)
#         #     for idx in range(w_loc, w_loc + self.w):
#         #         if len(self.table1[idx]) == 1:
#         #             q.append(idx)
#         #     while len(q) != 0:
#         #         i = q.pop(0)
#         #         if len(self.table1[i]) == 1:
#         #             x = list(self.table1[i])[0]
#         #             if x not in [tup[0] for tup in sigma]:
#         #                 sigma.append((x, i))
#         #                 hash_values = self.hash_all(x, self.w_hashes)
#         #                 for idx in range(len(self.w_hashes)):
#         #                     try:
#         #                         self.table1[w_loc + hash_values[idx]].remove(x)
#         #                     except KeyError:
#         #                         continue
#         #                     if len(self.table1[w_loc + hash_values[idx]]) == 1:
#         #                         q.append(w_loc + hash_values[idx])
#         print(len(sigma))
#         print(len(S))
#         return sigma if len(sigma) == len(S) else None
#
#     def assign(self, sigma):
#         for x, i in sigma:
#             hash_values = [self.hashes[func](x) for func in range(self.d)]
#             bit_values = [self.table2[hash_value] for hash_value in hash_values]
#             self.table2[i] = self.f(x) ^ np.bitwise_xor.reduce(bit_values)
#
#     def query(self, element):
#         hash_values = self.hash_all(element, self.w_hashes)
#         w_loc = self.slots[self.h_w(randint(0, 1000))]
#         slots_within_window = [w_loc + hash_val for hash_val in hash_values]
#         final_xor_values = [self.table2[slot] for slot in slots_within_window]
#         xor_result = np.bitwise_xor.reduce(final_xor_values)
#
#         return xor_result == 0
#
#     def calculate_false_positive_rate(self, test_set):
#         false_positives = 0
#
#         for element in test_set:
#             is_member = self.query(element)
#             # Check if element is a false positive.
#             if is_member and element not in self.w_table:
#                 false_positives += 1
#
#         # Determine false positive rate.
#         false_positive_rate = false_positives / len(test_set)
#         return false_positive_rate
#
#     @functools.lru_cache(maxsize=None)
#     def hash_all(self, t, hashes):
#         return [hashes[i](t) for i in range(len(hashes))]

# xf = XorFilter(elems=querylist[:100000], c=1.23, d=3, l=8)
# print(f"xor filter time: {xf.build_time}")
#
#
# xfsp = CoupledXorFilter(elems=querylist[:1200000], c=1.23, d=10, l=8, w=0.2)
# print(f"coupled xor filter time: {xfsp.build_time}")
# test_set = []
# for _ in range(10000):
#     test_set.append(''.join(random.choices(string.ascii_letters, k=10)))
#
# print(xfsp.calculate_false_positive_rate(test_set))

# figure, axes = plt.subplots(nrows=1, ncols=2)
# ax0, ax1 = axes
#
# c_vals = [1.23, 1.26, 1.29, 1.32, 1.35, 1.38]
# build_times_norm = [XorFilter(elems=querylist[:1200000], c=c, d=3, l=8).build_time for c in c_vals]
# build_times_coupled = [CoupledXorFilter(elems=querylist[:1200000], c=c, d=3, l=8, w=0.25).build_time for c in c_vals]
# ax0.plot(c_vals, build_times_norm)
# ax0.plot(c_vals, build_times_coupled)
# ax0.set_title("Build Time for XOR and Coupled XOR Filters (AOL Dataset)")
# ax0.set_xlabel("Filter Size / Dataset Size")
# ax0.set_ylabel("Build Time (s)")
# ax0.legend(labels=["xor filter", "coupled xor filter"], loc="lower right")
# ax0.plot(c_vals, build_times_norm)
# ax0.plot(c_vals, build_times_coupled)
