#! /usr/bin/env python
from itertools import pairwise

tests = [
    ([[0, 20], [-100000000, 10], [30, 40]], 100000030),
    ([[1, 2], [6, 10], [11, 15]], 9),
    ([[1, 5], [10, 20], [1, 6], [16, 19], [5, 11]], 19),
]


def sum_of_intervals(intervals):
    arr = [list(x) for x in sorted(intervals, key=lambda x: x[0])]
    new_arr = [arr[0]]
    cum_sum = 0
    for rng in arr[1:]:
        if rng[0] < new_arr[-1][1]:
            new_max = max(new_arr[-1][1], rng[1])
            new_arr[-1][1] = new_max
        else:
            cum_sum += new_arr[-1][1] - new_arr[-1][0]
            new_arr.append(rng)
    cum_sum += new_arr[-1][1] - new_arr[-1][0]
    return cum_sum


def test_int():
    for test_l, solution in tests:
        print(f"Test {test_l} solution {solution}")
        assert sum_of_intervals(test_l) == solution


if __name__ == "__main__":
    test_int()
