#! /usr/bin/env python
import numpy as np
import time


def find_smallest_order_n(rand_list):
    smallest = rand_list[0]
    for i in rand_list:
        if i < smallest:
            smallest = i
    return smallest


def find_smallest_order_n2(rand_list):
    candidate = rand_list[0]
    for i, item in enumerate(rand_list):
        for j in rand_list[i:]:
            if item > j:
                continue
        if item < candidate:
            candidate = item

    return candidate


total_elements = 10000
rand_list = np.random.randint(-100, 100, total_elements)
print(f"The list is has {total_elements} elements.")
init_time = time.time()
print(find_smallest_order_n(rand_list))
finish_t = time.time()
print(f"Execution time: {finish_t - init_time}")
init_time = time.time()
print(find_smallest_order_n2(rand_list))
finish_t = time.time()
print(f"Execution time N2: {finish_t - init_time}")
