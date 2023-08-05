# encoding: utf-8
"""
Solutions to the second hundred problems of Project Euler.
"""
# pylint: disable=line-too-long, bad-whitespace, invalid-name

import wrappy
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict


@wrappy.probe()
def euler_problem_101(coeffs=[(-1) ** _ for _ in range(0, 11)]):
    """
    https://projecteuler.net/problem=101
    """
    import numpy as np

    max_order = len(coeffs) - 1
    powers = np.array(
        [[_b**_p for _p in range(0, max_order + 1)] for _b in range(1, max_order + 2)]
    )
    points = powers.dot(coeffs)

    bops = [points[0]]
    for _order in range(1, max_order):
        _powers = powers[: (_order + 1), : (_order + 1)]
        _points = points[: (_order + 1)]
        _coeffs = np.concatenate(
            [
                np.linalg.inv(_powers).dot(_points),
                np.zeros(max_order - _order),
            ]
        )
        _fits = powers.dot(_coeffs)
        # only take the first "bad OP"
        for _p, _f in zip(points, _fits):
            if round(_p) != round(_f):
                bops.append(round(_f))
                break

    return bops


@wrappy.probe()
def euler_problem_102():
    """
    https://projecteuler.net/problem=102
    """
    import numpy as np

    """
    Idea: consider the vectors v1, v2, v3 where vi = [xi, yi].
    Any other point u can be writte as a linear combination of the vectors
    i.e. av1 + bv2 + cv3 = [xu, yu] where a + b + c = 1.
    So we have
    ax1 + bx2 + cx3 = xu
    ay1 + by2 + cy3 = yu
     a  +  b  +  c  = 1
    Which is a Ax=b problem.

    u is in the interior iff each of a, b, c is nonnegative.
    """
    with open("attachments/p102_triangles.txt", "r") as f:
        lines = [_ for _ in f.read().split("\n") if len(_) > 0]
        arrs = [list(map(int, _.split(","))) for _ in lines]

    interior_count = 0
    b = np.array([0, 0, 1])
    for _arr in arrs:
        _ax, _ay, _bx, _by, _cx, _cy = _arr
        _A = np.array(
            [
                [_ax, _bx, _cx],
                [_ay, _by, _cy],
                [1, 1, 1],
            ]
        )
        _x = np.linalg.inv(_A).dot(b)
        if (_x >= 0).all():
            interior_count += 1

    return interior_count


@wrappy.probe()
def euler_problem_103():
    """
    https://projecteuler.net/problem=103
    """

    @lru_cache(maxsize=int(1e6))
    def subset_sums(set_as_sorted_tuple):
        if not set_as_sorted_tuple:
            return {tuple([]): 0}

        arr = set_as_sorted_tuple
        if len(arr) == 1:
            return {tuple([]): 0, tuple(arr): sum(arr)}

        subsolution = subset_sums(arr[:-1])
        solution = subsolution.copy()
        for _tuple, _sum in subsolution.items():
            _new_tuple = tuple([*_tuple, arr[-1]])
            _new_sum = _sum + arr[-1]
            solution[_new_tuple] = _new_sum

        return solution

    def nonempty_subset_sums(set_as_sorted_tuple):
        solution = subset_sums(set_as_sorted_tuple).copy()
        keys_to_pop = []
        for _key in solution.keys():
            if len(_key) < 1:
                keys_to_pop.append(_key)

        for _key in keys_to_pop:
            solution.pop(_key)
        return solution

    def is_special_sum_set(set_as_sorted_tuple, verbose=False):
        # basic set check: no duplicates
        if len(set_as_sorted_tuple) != len(set(set_as_sorted_tuple)):
            return False

        subset_to_sum = nonempty_subset_sums(set_as_sorted_tuple)

        sums = set()
        size_to_sums = defaultdict(set)

        for _tuple, _sum in subset_to_sum.items():
            if _sum in sums:
                return False
            sums.add(_sum)
            size_to_sums[len(_tuple)].add(_sum)

        max_sum = 0
        for _size in sorted(size_to_sums.keys()):
            _sumset = size_to_sums[_size]
            if min(_sumset) <= max_sum:
                return False
            max_sum = max(_sumset)
            if verbose:
                print(_size, max_sum, _sumset)
        return True

    informed_guess = [20, *[20 + _ for _ in [11, 18, 19, 20, 22, 25]]]

    def search(guess_arr, low=-3, high=3):
        def subroutine(arr, perturb_idx):
            best_value = None
            best_arr = None
            for _diff in range(low, high + 1):
                arr[perturb_idx] += _diff

                if is_special_sum_set(tuple(sorted(arr))):
                    _value = sum(arr)
                    if best_value is None or _value < best_value:
                        best_value = _value
                        best_arr = arr[:]

                if perturb_idx < len(arr) - 1:
                    _value, _arr = subroutine(arr, perturb_idx=perturb_idx + 1)
                    if _value is not None and (
                        best_value is None or _value < best_value
                    ):
                        best_value = _value
                        best_arr = _arr[:]

                arr[perturb_idx] -= _diff
            return best_value, best_arr

        return subroutine(guess_arr, 0)

    result = search(informed_guess[:])
    assert is_special_sum_set(tuple(result[1]), verbose=True)
    return result


@wrappy.probe()
def euler_problem_104(bound=1000000, approx_thresh=500, verbose=False):
    """
    https://projecteuler.net/problem=104
    """
    import math

    cache = {0: 0, 1: 1, 2: 1}
    cache_last_10 = {}
    pandigital_set = set("123456789")

    def fibonacci(n):
        if n in cache:
            return cache[n]
        assert (
            isinstance(n, int) and approx_thresh >= n > 2
        ), f"Invalid Fibonacci index {n}"
        value = fibonacci(n - 2) + fibonacci(n - 1)
        cache[n] = value
        return value

    def fibonacci_last_10(n):
        if n < approx_thresh:
            return fibonacci(n)
        if n in cache_last_10:
            return cache_last_10[n]
        if n in cache:
            value = cache[n] % int(1e10)
            cache_last_10[n] = value
            return value
        value = (fibonacci_last_10(n - 2) + fibonacci_last_10(n - 1)) % int(1e10)
        cache_last_10[n] = value
        return value

    def begins_pandigitally(num):
        if num < int(1e8):
            return False

        return set(str(num)[:9]) == pandigital_set

    def ends_pandigitally(num):
        if num < int(1e8):
            return False

        return set(str(num)[-9:]) == pandigital_set

    approx_fibo, approx_factor = 0, (math.sqrt(5) + 1) / 2
    for k in tqdm(range(bound)):
        fibo_last_10 = fibonacci_last_10(k)
        if k < approx_thresh:
            fibo_first_10 = fibonacci(k)
        else:
            if k == approx_thresh:
                prev_fibo = fibonacci(approx_thresh - 1)
            else:
                prev_fibo = approx_fibo
            # keep the first 50 digits for precision
            while prev_fibo > 1e50:
                prev_fibo /= 10
            approx_fibo = round(prev_fibo * approx_factor)
            fibo_first_10 = approx_fibo

        begin_flag = begins_pandigitally(fibo_first_10)
        end_flag = ends_pandigitally(fibo_last_10)
        if verbose:
            if begin_flag:
                print(f"{k}th Fibonacci number begins pandigitally")
            if end_flag:
                print(f"{k}th Fibonacci number ends pandigitally")
        if begin_flag and end_flag:
            return k
