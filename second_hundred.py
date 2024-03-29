# encoding: utf-8
"""
Solutions to the second hundred problems of Project Euler.
"""
# pylint: disable=line-too-long, bad-whitespace, invalid-name

import wrappy
from tqdm import tqdm
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
    from subroutines import is_special_sum_set

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


@wrappy.probe()
def euler_problem_105():
    """
    https://projecteuler.net/problem=105
    """
    from subroutines import is_special_sum_set

    with open("attachments/p105_sets.txt", "r") as f:
        lines = [_ for _ in f.read().split("\n") if len(_) > 0]
        tuples = [tuple(sorted(set((map(int, _.split(",")))))) for _ in lines]

    total = sum([sum(_t) for _t in tqdm(tuples) if is_special_sum_set(_t)])
    return total


@wrappy.probe()
def euler_problem_106(n=12):
    """
    https://projecteuler.net/problem=106
    """
    from subroutines import Combination

    """
    Idea: the two subsets only need to be tested for equality when
    (1) they have the same number of elements
    (2) elements in one set are not pairwise greater to those in the other set

    The flip side of (2), where pairwise comparison works, is equivalent to
    walking down a grid from top left to bottom right without crossing the diagonal.
    """
    comb = Combination()

    def grid_walk(k):
        if k < 2:
            return 1

        arr = [1] * (k + 1)

        for _ in range(k):
            next_arr = [0] * (len(arr) - 1)
            next_arr[0] = arr[1]
            for i in range(1, len(next_arr)):
                next_arr[i] = next_arr[i - 1] + arr[i + 1]
            arr = next_arr
        return arr[-1]

    num_pairs_to_check = 0
    for i in range(1, (n // 2) + 1):
        _num_selections = comb.n_choose_k(n, 2 * i)
        _num_subselections = comb.n_choose_k(2 * i, i) // 2
        _num_subselections_to_check = _num_subselections - grid_walk(i)
        num_pairs_to_check += _num_selections * _num_subselections_to_check
    return num_pairs_to_check


@wrappy.probe()
def euler_problem_107():
    """
    https://projecteuler.net/problem=107
    """
    from subroutines import prim_mst

    with open("attachments/p107_network.txt", "r") as f:
        lines = [_ for _ in f.read().split("\n") if len(_) > 0]
        adj_list = [
            [(i, int(_)) for i, _ in enumerate(_line.split(",")) if _ != "-"]
            for _line in lines
        ]

    # note that the graph is undirected -> divide cost by 2 due to doublecounting
    full_cost = sum([sum([_[1] for _ in _adj]) for _adj in adj_list]) // 2
    mst = prim_mst(adj_list)
    mst_cost = sum([_[0] for _ in mst])
    return full_cost - mst_cost


@wrappy.probe()
def euler_problem_108(thresh=1000, prime_bound=int(1e3)):
    """
    https://projecteuler.net/problem=108
    """
    from subroutines import (
        all_primes_under,
        get_num_divisors,
        restore_from_factorization,
    )
    from datastruct import Heap
    from math import log10, sqrt
    from collections import defaultdict

    """
    Idea: without loss of generality, assume x <= y. So n < x <= y.
    Since 1/(2n) + 1/(2n) = 1/n, we have x <= 2n.
    1/x + 1/y = 1/n
    => yn + xn = xy
    => yn + xn - xy = 0
    => (n - x)(n - y) = n^2
    => (x - n)(y - n) = n^2
    So x can be any divisor of n^2 up to n. That is half the divisors other than n, and n itself.
    """
    sq_thresh = thresh * 2 - 1
    primes = all_primes_under(prime_bound)
    log_sq_thresh = log10(sq_thresh)

    heap = Heap([(log10(3) / log10(_p**2), _p, 2) for _p in primes], minheap=False)

    log_accumulated = 0
    factors = defaultdict(int)
    while log_accumulated < log_sq_thresh:
        _effi, _prime, _order = heap.extract()
        _incre = _effi * log10(_prime**2)
        log_accumulated += _incre
        factors[_prime] += 2
        heap.insert(
            (
                (log10(_order + 3) - log10(_order + 1)) / (log10(_prime**2)),
                _prime,
                _order + 2,
            )
        )
    primes_to_consider = [_p for _p in primes if _p <= max(factors.keys())]
    print(restore_from_factorization(factors), factors, get_num_divisors(factors))

    def search(guess_factorization, low=-2, high=2):
        def subroutine(factorization, perturb_idx):
            perturb_prime = primes_to_consider[perturb_idx]
            best_value = None
            best_factorization = None
            for _diff in range(low, high + 1, 2):
                if factorization[perturb_prime] + _diff < 0:
                    continue
                factorization[perturb_prime] += _diff

                if get_num_divisors(factorization) > sq_thresh:
                    _value = restore_from_factorization(factorization)
                    if best_value is None or _value < best_value:
                        best_value = _value
                        best_factorization = factorization.copy()

                if perturb_idx < len(primes_to_consider) - 1:
                    _value, _fac = subroutine(
                        factorization, perturb_idx=perturb_idx + 1
                    )
                    if _value is not None and (
                        best_value is None or _value < best_value
                    ):
                        best_value = _value
                        best_factorization = _fac.copy()

                factorization[perturb_prime] -= _diff
            return best_value, best_factorization

        return subroutine(guess_factorization, 0)

    best_value, factors = search(factors)
    return int(sqrt(best_value)), factors, get_num_divisors(factors)


@wrappy.probe()
def euler_problem_110(thresh=4000000, prime_bound=int(1e3)):
    """
    https://projecteuler.net/problem=110
    """
    # exact same method as problem 108
    return euler_problem_108(thresh=thresh, prime_bound=prime_bound)


@wrappy.probe()
def euler_problem_111(num_digits=10):
    """
    https://projecteuler.net/problem=111
    """
    """
    Idea: pre-compute primes up to the square root of the maximal number.
    For each d, start trying from the most repeats. Checking prime is cheap.
    """
    from subroutines import (
        all_primes_under,
        is_prime_given_primes,
        permutations_by_insertion,
    )

    basic_primes = all_primes_under(int(2 * 10 ** (num_digits / 2)))

    def generate_filler_digits(repeated_digit, num_to_generate):
        assert isinstance(num_to_generate, int) and num_to_generate >= 0
        assert isinstance(repeated_digit, int) and 0 <= repeated_digit <= 9
        if num_to_generate == 0:
            yield []
        else:
            for _digits in generate_filler_digits(repeated_digit, num_to_generate - 1):
                for i in range(0, 10):
                    if i == repeated_digit:
                        continue
                    else:
                        yield [*_digits, i]

    solutions = {}
    for _d in range(0, 10):
        for _M in range(num_digits - 1, 1, -1):
            _base = [_d] * _M
            _seen = set()
            _qualified = set()
            for _filler in generate_filler_digits(_d, num_digits - _M):
                for _permu in permutations_by_insertion(_base[:], _filler):
                    _num = int("".join(map(str, _permu)))
                    # filter duplicates
                    if _num in _seen:
                        continue
                    _seen.add(_num)
                    # ignore numbers with leading 0 or already seen
                    if len(str(_num)) < num_digits:
                        continue
                    if is_prime_given_primes(_num, basic_primes):
                        _qualified.add(_num)
            # terminate if found valid M value
            if _qualified:
                solutions[_d] = dict(
                    M=_M, N=len(_qualified), S=sum(_qualified), primes=_qualified
                )
                break

    return solutions


@wrappy.probe()
def euler_problem_112(target=0.99):
    """
    https://projecteuler.net/problem=112
    """
    from subroutines import BouncyNumberHelper

    numer, denom = 0, 99
    for _digits in range(3, 10):
        for _first in range(1, 10):
            _numer_inc = BouncyNumberHelper.bouncy_numbers_given_num_digits_and_first(
                _digits, _first
            )
            _denom_inc = 10 ** (_digits - 1)
            if (numer + _numer_inc) / (denom + _denom_inc) > target:
                _start = _first * 10 ** (_digits - 1)
                _end = (_first + 1) * 10 ** (_digits - 1)
                for _num in range(_start, _end):
                    if BouncyNumberHelper.is_bouncy(_num):
                        numer += 1
                    denom += 1
                    if numer / denom >= target:
                        return _num
                raise ValueError(f"Expected target number between {_start} and {_end}")
            else:
                numer += _numer_inc
                denom += _denom_inc

    return None


@wrappy.probe()
def euler_problem_113(digit_limit=100):
    """
    https://projecteuler.net/problem=113
    """
    from subroutines import BouncyNumberHelper

    total = 99
    for _digits in range(3, digit_limit + 1):
        for _first in range(1, 10):
            total += 10 ** (_digits - 1)
            total -= BouncyNumberHelper.bouncy_numbers_given_num_digits_and_first(
                _digits, _first
            )

    return total


@wrappy.probe()
def euler_problem_114(m=3, n=50):
    """
    https://projecteuler.net/problem=114
    """
    """
    Idea: this is classic dynamic programming.
    Subproblem arrangements can end in a red or black block.
    We then attach red or black blocks in the next unit.
    Red blocks can only happen in groups of m=3, so look 3 units behind for a black end.
    """
    from subroutines import block_tiling_flexible_1d

    end_in_red, end_in_black = block_tiling_flexible_1d(m, n)
    return end_in_red[-1] + end_in_black[-1]


@wrappy.probe()
def euler_problem_115(m=50, n_guess=200, target=int(1e6)):
    """
    https://projecteuler.net/problem=115
    """
    from subroutines import block_tiling_flexible_1d

    end_in_red, end_in_black = block_tiling_flexible_1d(m, n_guess)

    for i in range(n_guess):
        if end_in_red[i] + end_in_black[i] > target:
            return i + 1


@wrappy.probe()
def euler_problem_116(m_values=(2, 3, 4), n=50):
    """
    https://projecteuler.net/problem=116
    """
    from subroutines import block_tiling_fixed_1d

    total = 0
    for _m in m_values:
        # un-count the all-black tiling
        _end_in_red, _end_in_black = block_tiling_fixed_1d(_m, n)
        total += _end_in_red[-1] + _end_in_black[-1] - 1
    return total


@wrappy.probe()
def euler_problem_117(m_values=(2, 3, 4), n=50):
    """
    https://projecteuler.net/problem=117
    """
    from subroutines import block_tiling_multifixed_1d

    end_in_red, end_in_black = block_tiling_multifixed_1d(m_values, n)
    return end_in_red[-1] + end_in_black[-1]


@wrappy.probe()
def euler_problem_118(all_digits="123456789"):
    """
    https://projecteuler.net/problem=118
    """
    """
    Idea: use two lookups using sorted digits as the key, e.g. '12359'.
    (1) digits-to-primes lookup for finding prime candidiates.
    (2) digits-to-sets lookup for finding subsolutions.
    We represent each solution as comma-separated sorted array to deduplicate.
    For example, {2, 47, 5, 89, 631} -> '2,5,47,89,631'
    """
    from collections import defaultdict
    from subroutines import (
        all_primes_under,
        is_prime_given_primes,
        generate_all_partitions_from_list,
        permutations_from_list,
    )

    all_digits = "".join(sorted(list(all_digits)))
    basic_primes = all_primes_under(32 * int(1e3))
    digits_to_primes = defaultdict(list)
    for _l, _ in tqdm(
        generate_all_partitions_from_list(list(all_digits)),
        total=int(2 ** len(all_digits)),
    ):
        _key = "".join(sorted(_l))
        for _larr in permutations_from_list(_l):
            if not _larr:
                continue
            _lnum = int("".join(_larr))
            if _lnum < 2:
                continue
            if is_prime_given_primes(_lnum, basic_primes):
                digits_to_primes[_key].append(_lnum)

    digits_to_solutions = defaultdict(set)

    def subsolutions(digits):
        # repeated case: lookup
        if digits in digits_to_solutions:
            return digits_to_solutions[digits]

        # base case: 1 digit only, no partition needed
        if len(digits) == 1:
            digits_to_solutions[digits] = [str(_p) for _p in digits_to_primes[digits]]
            return digits_to_solutions[digits]

        solutions = set()
        # recursive case: partition to two groups
        for _l, _r in generate_all_partitions_from_list(list(digits)):
            _l, _r = "".join(_l), "".join(_r)
            if not _r:
                continue
            if not _l:
                for _rp in digits_to_primes[_r]:
                    solutions.add(str(_rp))
            # left group looks for subsolution
            for _lsol in subsolutions(_l):
                _larr = list(map(int, _lsol.split(",")))
                # right group looks for primes
                for _rp in digits_to_primes[_r]:
                    _sol = sorted([*_larr, _rp])
                    solutions.add(",".join(map(str, _sol)))

        digits_to_solutions[digits] = solutions
        return digits_to_solutions[digits]

    return subsolutions(all_digits)


@wrappy.probe()
def euler_problem_119(target_idx=30, base_bound=500, power_bound=20):
    """
    https://projecteuler.net/problem=119
    """
    """
    Idea: just try combinations of base numbers and powers.
    Note that if a number has more digits than the base number,
    it is very unlikely for the digits to sum up to the base.
    """
    # get as many qualified numbers as target_idx
    # so the maximum so far is an upper bound of the target number
    qualified = []

    for _b in range(1, base_bound + 1):
        for _p in range(2, power_bound + 1):
            _num = _b**_p
            if _num < 10:
                continue
            _sum = sum(list(map(int, list(str(_num)))))
            if _sum == _b:
                qualified.append(_num)

    # try bases and powers up to the target number
    # note that the base is at most 9 * number_of_digits
    num_limit = sorted(qualified)[target_idx - 1]
    base_limit = len(str(num_limit)) * 9
    qualified.clear()

    for _b in tqdm(range(2, base_limit + 1)):
        _p = 1
        while True:
            _p += 1
            _num = _b**_p
            if _num < 10:
                continue
            if _num > num_limit:
                break
            _sum = sum(list(map(int, list(str(_num)))))
            if _sum == _b:
                qualified.append((_num, _b, _p))

    return sorted(qualified)


@wrappy.probe()
def euler_problem_120(low=3, high=1000):
    """
    https://projecteuler.net/problem=120
    """
    """
    Idea: look into the pattern of (a - 1) ^ p and (a + 1) ^ p mod a ^ 2.
    We can eliminate any 2nd order term on sight, i.e. it's at most 1st order.
    Example:
     p | (a-1)^p | (a+1)^p | (a-1)^p + (a+1)^p
    ---|---------|---------|-------------------
     1 |   a - 1 |   a + 1 | 2 (a)
     2 | -2a + 1 |  2a + 1 | 2
     3 |  3a - 1 |  3a + 1 | 2 (3a)
     4 | -4a + 1 |  4a + 1 | 2
    Note the 2 * (odd) multiplier before a. This series is 4n + 2, which implies
    that the max remainder has to be (the greatest even number below a) times a.
    This is because
    - 4 is coprime to all odd numbers
    - for a = 0 mod 4, 4n + 2 will reach a - 2.
    - for a = 2 mod 4, 4n + 2 will reach 2a - 2.
    """
    total = 0
    for _num in range(low, high + 1):
        _max_l_even = _num - (2 if _num % 2 == 0 else 1)
        total += _max_l_even * _num
    return total


@wrappy.probe()
def euler_problem_121(num_turns=15):
    """
    https://projecteuler.net/problem=121
    """
    from math import floor
    from subroutines import generate_combinations_from_integer_range, factorial

    """
    Idea: given n turns there can be up to (n - 1) // 2 reds.
    Count the number of events given which turns got reds.
    """
    events_per_turn = [(i + 1) for i in range(1, num_turns + 1)]
    total_events = factorial(num_turns + 1)

    # pre-count the event of all blues
    total_winning_events = 1
    for _reds in range(1, (num_turns - 1) // 2 + 1):
        for _arr in generate_combinations_from_integer_range(
            elements=_reds, low=0, high=num_turns - 1
        ):
            _events = 1
            for _idx in _arr:
                _events *= events_per_turn[_idx] - 1
            total_winning_events += _events

    return floor(total_events / total_winning_events)


@wrappy.probe()
def euler_problem_122(max_order=200):
    """
    https://projecteuler.net/problem=122
    """
    """
    Textbook dynamic programming except that subproblem solutions need to
    account for all exponents involved in multiplications.
    """
    solution = [None] * (max_order + 1)
    solution[:3] = [[set()], [set()], [{2}]]
    for _k in tqdm(range(3, max_order + 1)):
        _min_muls = []
        for _m in range(1, (_k // 2) + 1):
            for _mul_set_l in solution[_m]:
                for _mul_set_r in solution[_k - _m]:
                    _mul_set = {_k}
                    _mul_set.update(_mul_set_l)
                    _mul_set.update(_mul_set_r)

                    if not _min_muls:
                        _min_muls.append(_mul_set)
                    elif len(_mul_set) == len(_min_muls[0]):
                        _min_muls.append(_mul_set)
                    elif len(_mul_set) < len(_min_muls[0]):
                        _min_muls.clear()
                        _min_muls.append(_mul_set)
        solution[_k] = _min_muls

    return solution[1 : (max_order + 1)]


@wrappy.probe()
def euler_problem_123(thresh=int(1e10)):
    """
    https://projecteuler.net/problem=123
    """
    from math import sqrt
    from subroutines import all_primes_under

    """
    Idea: look into the pattern of (a - 1) ^ p and (a + 1) ^ p mod a ^ 2.
    We can eliminate any 2nd order term on sight, i.e. it's at most 1st order.
    Example:
     p | (a-1)^p | (a+1)^p | (a-1)^p + (a+1)^p
    ---|---------|---------|-------------------
     1 |   a - 1 |   a + 1 | 2 (a)
     2 | -2a + 1 |  2a + 1 | 2
     3 |  3a - 1 |  3a + 1 | 2 (3a)
     4 | -4a + 1 |  4a + 1 | 2
    So the reminder is (2p % a) * a when p is odd.
    """
    primes = all_primes_under(int(10 * sqrt(thresh)))

    for i, _a in enumerate(primes):
        if i % 2 == 1:
            continue
        _p = i + 1
        _remainder = ((2 * _p) % _a) * _a
        if _remainder > thresh:
            return _p, _a, _remainder
    return None


@wrappy.probe()
def euler_problem_124(limit=100000, idx=10000):
    """
    https://projecteuler.net/problem=124
    """
    from subroutines import Factorizer

    fac = Factorizer()

    rad_n_arr = [(1, 1)]
    for _n in range(2, limit + 1):
        _rad = 1
        for _p in fac.factorize(_n).keys():
            _rad *= _p
        rad_n_arr.append((_rad, _n))

    return sorted(rad_n_arr)[idx - 1]


@wrappy.probe()
def euler_problem_125(bound=int(1e8)):
    """
    https://projecteuler.net/problem=125
    """
    from math import ceil, sqrt
    from subroutines import is_a_palindrome

    """
    Idea: it's easy to check for palindrome and there seems to be no
    mathematical connection between palindromes and consecutive square sums.
    So pre-compute cumulative sums to efficiently find consecutive ones.
    """
    base_bound = 2 * ceil(sqrt(bound))
    squares = [_**2 for _ in range(0, base_bound + 1)]
    cumulative_squares = squares[:]
    for i in range(1, base_bound + 1):
        cumulative_squares[i] += cumulative_squares[i - 1]

    # some palindromes can be written as different sums!
    # this means we need to handle double counting
    palindrome_to_bases = {}
    for i in tqdm(range(0, base_bound - 1)):
        for j in range(i + 2, base_bound + 1):
            _candidate = cumulative_squares[j] - cumulative_squares[i]
            if _candidate > bound:
                break
            if is_a_palindrome(_candidate) and _candidate < bound:
                if _candidate not in palindrome_to_bases:
                    palindrome_to_bases[_candidate] = []
                palindrome_to_bases[_candidate].append(list(range(i + 1, j + 1)))

    return palindrome_to_bases


@wrappy.probe()
def euler_problem_126(layer_size_limit=30000):
    """
    https://projecteuler.net/problem=126
    """
    """
    Deprecated idea: start from a 1-by-1-by-1 cube.
    Consider the cube in x, y, z dimensions with length (a, b, c) where,
    without loss of generality, a >= b >= c.
    Imagine stretching the cube in the three dimensions.
    The same stretching applies to any layers the cuboid has -- the cubes
    per unit stretched along an axis is the area obtained by projecting
    the cuboid in that axis.
    """
    """
    Idea: divide the layered cuboid into different kinds of pieces:
    (1) starting cuboid
    (2) extra cuboids "grown" from the six faces
    (3) triangular shapes interpolated from (2)
    (4) tetrahedrons interpolated from 3
    Then we have a formula of the total volume.

    def volume(a, b, c, k):
        # base volume at k = 0
        vol = a * b * c
        # extra cuboids from 6 faces: starting at k = 1
        if k >= 1:
            for _face in [a * b, b * c, c * a]:
                vol += 2 * k * _face
        # extra triangulars from 12 sides: starting at k = 2
        if k >= 2:
            for _side in [a, b, c]:
                vol += 4 * _side * ((k - 1) * k / 2)
        # extra tetrahedrons from 8 vertices: starting at k = 3
        if k >= 3:
            vol += 8 * ((k - 2) * (k - 1) * k / 6)
        return round(vol)

    Taking the diff between volume(a, b, c, k) and volumn(a, b, c, k - 1)
    gives us the formula for layer_size(a, b, c, k).
    """

    def layer_size(a, b, c, k):
        size = 2 * (a * b + b * c + c * a)
        size += (4 * k - 4) * (a + b + c)
        size += 4 * (k - 1) * (k - 2)
        return size

    layer_size_to_specs = defaultdict(list)
    dim_limit = layer_size_limit // 4

    for _a in tqdm(range(1, dim_limit)):
        _b_bound = min(_a, layer_size_limit // (2 * _a))
        for _b in range(1, _b_bound + 1):
            _c_bound = min(_b, (layer_size_limit - 2 * _a * _b) // (2 * (_a + _b)))
            for _c in range(1, _c_bound + 1):
                for _k in range(1, int(1e6)):
                    _layer_size = layer_size(_a, _b, _c, _k)
                    if _layer_size > layer_size_limit:
                        break
                    layer_size_to_specs[_layer_size].append((_a, _b, _c, _k))

    return sorted(layer_size_to_specs.items(), key=lambda t: t[0])


@wrappy.probe()
def euler_problem_128(terms=2000):
    """
    https://projecteuler.net/problem=128
    """
    """
    Idea: consider traversing the numbers by their values.
    1 -> 2 -> 3 -> ...
    Use a index schema (r, v, t) or "ring, vertex, tile" where
    - ring stands for which ring the number is in
    - vertex stands for the last vertex before the number on the same ring
    - tile stands for the steps from the vertex to the number.
    Then 1 is written as (0, 0, 0), 8 is written as (2, 0, 0),
    9 is (2, 0, 1), 14 is (2, 3, 0), 25 is (3, 1, 2).

    This is true for (at least) r >= 3:
    If a number is in the middle of a ring, two of the neighbor diffs will be 1.
    Two of the remaining four diffs will have to be even, no matter whether t is 0.
    So the only candidates with three prime diffs will at the start or end of a ring.
    We can derive a formula of these numbers and the diffs.
    """
    from subroutines import HexagonRingsNeighborFinder, Factorizer

    fac = Factorizer()
    finder = HexagonRingsNeighborFinder()
    qualified = []

    def ring_start(r):
        return round(2 + 6 * (r - 1) * r / 2)

    def ring_end(r):
        return round(1 + 6 * (r + 1) * r / 2)

    def diffs_ring_start(r):
        ring = 6 * r
        yield ring - 1
        yield ring + 1
        yield 12 * r + 5

    def diffs_ring_end(r):
        ring = 6 * r
        yield ring - 1
        yield ring + 5
        yield 12 * r - 7

    def full_check(num):
        prime_diffs = 0
        for _nbr in finder.all_neighbors(num):
            _diff = abs(_nbr - num)
            if _diff != 1 and fac._least_divisor(_diff) == _diff:
                prime_diffs += 1
        return prime_diffs == 3

    # brute force for r < 3
    for _num in range(1, 20):
        if full_check(_num):
            qualified.append(_num)

    # check ring ends for r >= 3
    r = 3
    with tqdm(total=terms) as pbar:
        pbar.update(len(qualified))
        while len(qualified) < terms:
            _flag_start, _flag_end = True, True
            for _num in diffs_ring_start(r):
                if not fac.is_prime(_num):
                    _flag_start = False
                    break
            for _num in diffs_ring_end(r):
                if not fac.is_prime(_num):
                    _flag_end = False
                    break
            if _flag_start:
                qualified.append(ring_start(r))
                pbar.update(1)
            if _flag_end:
                qualified.append(ring_end(r))
                pbar.update(1)
            r += 1

    return qualified[terms - 1]


@wrappy.probe()
def euler_problem_145(max_digits=9):
    """
    https://projecteuler.net/problem=145
    """
    from subroutines import generate_combinations_from_element_wise_choices

    """
    Idea: denote the digits as d1, d2, ..., dn.
    Last digit of the sum is d1 + dn and must be odd.
    This means exactly one between d1 and dn is odd.

    Let's call any digit sum > 10 an "advancement".
    If n is even, no digit gets 2x.
    There cannot be any advancement because the last digit must be (odd, even)
    so the first digit (except a possible leading 1) cannot get any advancement.

    If n is odd, there will be a digit adding to itself. Let's call it X.
      E O X  E O
      O E X  O E
    1?O?O?2X?O?O
    The 2X must get an advancement, but then it implies an advancement on the first digit after the possible leading 1.
    So for n=5 there are no reversible numbers. However there can be for n=7.
    """

    class C:
        # each tuple: (code, combinations, odd, adv)
        # with even: count by 0, 2, 4, 6, 8 in that order
        # with odd only: count by 1, 3, 5, 7, 9 in that order
        # (odd, even), no adv, last digit (no zero)
        OEnl = ("OEnl", 0 + 4 + 3 + 2 + 1, 1, 0)
        # (odd, even), adv, last digit (no zero)
        OEal = ("OEal", 0 + 1 + 2 + 3 + 4, 1, 1)
        # (odd, even), no adv, not last digit
        # multiply by 2 for the (even, odd) exchange
        OEn = ("OEn", 2 * (5 + 4 + 3 + 2 + 1), 1, 0)
        # (odd, even), adv, not last digit
        # multiply by 2 for the (even, odd) exchange
        OEa = ("OEa", 2 * (0 + 1 + 2 + 3 + 4), 1, 1)
        # (even, even), no adv, not last digit
        EEn = ("EEn", 5 + 4 + 3 + 2 + 1, 0, 0)
        # (even, even), adv, not last digit
        EEa = ("EEa", 0 + 1 + 2 + 3 + 4, 0, 1)
        # (odd, odd), no adv, not last digit
        OOn = ("OOn", 4 + 3 + 2 + 1 + 0, 0, 0)
        # (even, even), adv, not last digit
        OOa = ("OOa", 1 + 2 + 3 + 4 + 5, 0, 1)
        # (x, x), no adv: 0, 1, 2, 3, 4
        XXn = ("XXn", 5, 0, 0)
        # (x, x), adv: 5, 6, 7, 8, 9
        XXa = ("XXa", 5, 0, 1)

    def get_sequence_of_possibilities(num_digits):
        num_nonx = num_digits // 2
        at_x = [[C.XXn, C.XXa]] if num_digits % 2 == 1 else []
        after_x = [
            *[[C.OEn, C.OEa, C.EEn, C.EEa, C.OOn, C.OOa] for _ in range(num_nonx - 1)],
            [C.OEnl, C.OEal],
        ]
        return [*at_x, *after_x]

    def valid_combinations(sequence):
        count = sequence[-1][1]
        for i in range(len(sequence) - 1):
            _step_count = sequence[i][1]
            _odd = sequence[i][2]
            _next_adv = sequence[i + 1][3]
            if _odd and _next_adv:
                return 0
            elif not _odd and not _next_adv:
                return 0
            else:
                count *= _step_count

        # count twice for each pair
        return count * 2

    total = 0
    for _num_digits in range(2, max_digits + 1):
        _subtotal = 0
        _seq_choices = get_sequence_of_possibilities(_num_digits)
        for _half_seq in generate_combinations_from_element_wise_choices(_seq_choices):
            _after_x_idx = 1 if _half_seq[0][0].startswith("XX") else 0
            _before_x = [
                tuple([_[0], 1, _[2], _[3]]) for _ in _half_seq[_after_x_idx:][::-1]
            ]
            _seq = [*_before_x, *_half_seq]

            _seqtotal = valid_combinations(_seq)
            if _seqtotal > 0:
                print([_[0] for _ in _seq], _seqtotal)
            _subtotal += _seqtotal
        total += _subtotal
        print(_num_digits, _subtotal)
    return total
