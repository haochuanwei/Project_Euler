"""
Subroutines that may get used repeatedly across different problems.
"""
# pylint: disable=line-too-long, bad-whitespace
import wrappy
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict


def least_divisor(num, floor=2):
    """
    Find the least divisor of a number, above some floor.
    """
    assert num >= floor
    trial = floor
    while num % trial != 0:
        trial += 1
    return trial


def has_nontrivial_divisor(num):
    """
    Determines if a number has a nontrivial divisor.
    """
    divisor = least_divisor(num)
    return bool(divisor < num)


def is_a_palindrome(num):
    """
    Determines if a number is a palindrome.
    """
    assert isinstance(num, int)
    str_form = str(num)
    n_digits = len(str_form)
    for k in range(0, (n_digits + 1) // 2):
        if str_form[k] != str_form[-1 - k]:
            return False
    return True


class Factorizer:  # pylint: disable=too-few-public-methods
    """
    A factorizer that makes use of multiple data structures.
    This is intended to be efficient for repeated factorizations.
    """

    def __init__(self, bound=10**6):
        """
        Initialize a cache of factorizations and precompute primes.
        bound -- the bound of numbers that the factorizer is expected to deal with.
        """
        self.bound = bound
        self.cache = {}
        self._update_primes()

    def _check_bound(self, num):
        if num > self.bound:
            print(
                "{0} exceeded the expected bound {1}. Updating known prime numbers.".format(
                    num, self.bound
                )
            )
            self.bound = num * 2
            self._update_primes()

    def _update_primes(self):
        """
        Update the list and set of primes, up to some bound.
        """
        from math import ceil, sqrt

        # we only need primes up to sqrt(bound) because if none of those primes divide a number under bound, then bound must be prime
        if hasattr(self, "list_primes") and self.list_primes[-1] ** 2 > self.bound:
            return
        self.list_primes = all_primes_under(ceil(sqrt(self.bound)))
        self.set_primes = set(self.list_primes)

    def _least_divisor(self, num):
        """
        Find the least divisor of a number.
        """
        self._check_bound(num)
        for _p in self.list_primes:
            if num % _p == 0:
                return _p
        # if none of the primes divide num, then num is a prime
        return num

    def factorize(self, num):
        """
        Factorize a number.
        """
        from collections import defaultdict

        self._check_bound(num)
        # case 0: num is too small
        if num < 2:
            return {}
        # case 1: num is already factorized
        if num in self.cache.keys():
            return self.cache[num]
        # case 2: num is a prime
        if num in self.set_primes:
            _factorization = {num: 1}
            self.cache[num] = _factorization
            return _factorization
        # common case: num is a composite number
        divisor = self._least_divisor(num)
        factor = int(num / divisor)
        _factorization = defaultdict(int)
        _factorization.update(self.factorize(factor))
        _factorization[divisor] += 1
        self.cache[num] = _factorization
        return _factorization


def restore_from_factorization(factorization):
    """
    Restore the original number from a factorization.
    """
    retval = 1
    for _base, _power in factorization.items():
        retval *= int(_base) ** int(_power)
    return retval


def get_num_divisors(factorization):
    """
    Determine the number of different divisors given a factorization.
    """
    from functools import reduce

    powers = list(factorization.values())
    num_divisors = reduce(lambda x, y: x * y, [_p + 1 for _p in powers])
    return num_divisors


def get_all_divisors(factorization):
    """
    Get all the divisors of a number given its factorization.
    """
    divisors = [1]
    for _base, _power in factorization.items():
        divisors = [
            _div * (_base**_p) for _p in range(0, _power + 1) for _div in divisors
        ]
    return divisors


def get_sum_proper_divisors(factorization):
    """
    Determine the sum of proper divisors given a factorization.
    """
    sum_divisors = 1
    original_number = 1
    for _base, _power in factorization.items():
        factors = [_base**k for k in range(0, _power + 1)]
        sum_divisors *= sum(factors)
        original_number *= _base**_power
    return sum_divisors - original_number


class DigitwiseInteger:
    """
    Integer that is represented by the value on each digit.
    """

    def __init__(self, num):
        """
        Initialize from a usual integer.
        """
        from collections import defaultdict

        self.order_dict = defaultdict(int)
        _digit = 0
        _value = num
        if _value == 0:
            self.order_dict[0] = 0
        while _value != 0:
            self.order_dict[_digit] = _value % 10
            _digit += 1
            _value = _value // 10

    def reconstruct(self):
        """
        Reconstruct the original number.
        """
        retval = 0
        for _digit, _value in self.order_dict.items():
            retval += _value * (10**_digit)
        return retval

    def __consistency_check(self):
        assert DigitwiseInteger(self.reconstruct()).reconstruct() == self.reconstruct()

    def multiply_by_constant(self, multiplier, in_place=False, check_result=False):
        """
        Subroutine to multiply a number by multiplier.
        Eg. {0: 1, 1: 3, 2: 6} stands for 1*10^1 + 3*10^1 + 6*10^2 = 631.
        """
        from collections import defaultdict

        def correct_digits(order_dict):
            """
            Promote digits with value greater than or equal to 10.
            """
            ret_dict = defaultdict(int)
            digits = sorted(order_dict.keys())
            # check digits from low to high
            for _digit in digits:
                _value = order_dict[_digit]
                # pickup value
                ret_dict[_digit] += _value
                # promote if appropriate
                if ret_dict[_digit] >= 10:
                    ret_dict[_digit + 1] += ret_dict[_digit] // 10
                    ret_dict[_digit] = ret_dict[_digit] % 10
            return ret_dict

        # perform calculation digit-wise
        ret_dict = defaultdict(int)
        for _key, _value in self.order_dict.items():
            multiplied = _value * multiplier
            shift = 0
            while multiplied > 0 or shift == 0:
                ret_dict[_key + shift] += multiplied % 10
                multiplied = multiplied // 10
                shift += 1
        # promote digits that have value greater than 10
        ret_dict = correct_digits(ret_dict)
        if in_place:
            self.order_dict = defaultdict(int)
            self.order_dict.update(ret_dict)
            if check_result:
                self.__consistency_check()
        return ret_dict


def factorial(num):
    """
    Factorial.
    """
    assert isinstance(num, int) and num >= 0
    if num == 0:
        return 1
    return num * factorial(num - 1)


def is_prime_given_factorization(factorization):
    """
    Given a factorization dict, determine if the original number is a prime.
    """
    if len(factorization.keys()) == 1:
        if list(factorization.values())[0] == 1:
            return True
    return False


def is_prime_given_primes(num, primes):
    """
    Determine if a number is prime, given an ascending list of prime numbers below its square root.
    """
    from math import floor, sqrt

    assert isinstance(num, int) and num >= 2
    assert primes[-1] >= floor(
        sqrt(num)
    ), f"Prime limit {primes[-1]} is insufficient for {num}"
    for _p in primes:
        if _p > floor(sqrt(num)):
            break
        if num % _p == 0:
            return False
    return True


def all_primes_under(bound):
    """
    Compute all the prime numbers below a bound.
    """

    def is_prime_with_cache(num, cache):
        """
        This is a subroutine for dynamic programming.
        Given a cache of primes below the square root of a number, determine if it is prime.
        The cache must be of ascending order.
        """
        from math import sqrt, ceil

        for _p in cache:
            if _p > ceil(sqrt(num)):
                break
            if num % _p == 0:
                return False
        cache.append(num)
        return True

    # use a list for keeping primes in ascending order
    cache_primes = []
    for candidate in tqdm(range(2, bound), desc=f"Calculating primes under {bound}"):
        is_prime_with_cache(candidate, cache_primes)
    return cache_primes[:]


def is_m_to_n_pandigital(num, bound_m, bound_n):
    """
    Determine if a number is m-to-n pandigital.
    """
    digit_count = dict()
    list_form = list(str(num))
    for _digit in list_form:
        # return early if any digit shows up more than once
        if _digit in digit_count.keys():
            return False
        digit_count[_digit] = 1
    target_count = dict()
    for _d in range(bound_m, bound_n + 1):
        target_count[str(_d)] = 1
    # compare two sets
    if digit_count == target_count:
        return True
    return False


def two_sum(arr, num):
    """
    The two-sum problem where the input array is already in set form.
    """
    combinations = []
    assert isinstance(arr, set)
    for term_a in arr:
        term_b = num - term_a
        if term_b >= term_a and term_b in arr:
            combinations.append((term_a, term_b))
    return combinations


def is_triangular(num):
    """
    Determine if a number is of the form (1/2)n(n+1).
    """
    from math import floor, sqrt

    assert isinstance(num, int) and num > 0
    near_sqrt = floor(sqrt(2 * num))
    return bool(int((1 / 2) * near_sqrt * (near_sqrt + 1)) == num)


def permutations_m_to_n_str(bound_m, bound_n):
    """
    Get all permutations of digits between m and n, in string form.
    Example:
    permutations_m_to_n_str(1, 3) -> ['123', '132', '213', '231', '312', '321']
    """

    def add(perms, new_digit):
        """
        Add a digit to existing permutations.
        Assumes that all existing permutations have the same length.
        """
        # base case: no permutation so far
        if not perms:
            return [new_digit]
        # common case
        perm_length = len(perms[0])
        retlist = []
        for _perm in perms:
            new_perms = [
                (_perm[:i] + new_digit + _perm[i:]) for i in range(0, perm_length)
            ]
            new_perms.append(_perm + new_digit)
            retlist += new_perms
        return retlist

    permutations = []
    for _d in range(bound_m, bound_n + 1):
        permutations = add(permutations, str(_d))
    return permutations


def permutations_from_list(arr):
    """
    Get all permutations of array elements.
    Example:
    permutations_from_list([1, 2, 3]) -> [1, 2, 3], [1, 3, 2], [2, 1, 3], ...
    """
    if not arr:
        yield []
    elif len(arr) == 1:
        yield arr
    else:
        for _ in permutations_from_list(arr[:-1]):
            for i in range(0, len(_) + 1):
                yield [*_[:i], arr[-1], *_[i:]]


def permutations_by_insertion(start_arr, insert_arr):
    """
    Get all permutations of array elements, starting from a given partial permutation.
    Example:
    [1, 2, 3], [4, 5] -> [1, 2, 3, 4, 5], [1, 2, 3, 5, 4], [1, 2, 4, 3, 5], ...
    where 1, 2, and 3 always show up in the original order.
    """
    assert start_arr
    if not insert_arr:
        yield start_arr[:]
    else:
        for _ in permutations_by_insertion(start_arr, insert_arr[:-1]):
            for i in range(0, len(_) + 1):
                yield [*_[:i], insert_arr[-1], *_[i:]]


def get_triangulars(num):
    """
    Get the first n triangular numbers.
    """
    return [int(i * (i + 1) / 2) for i in range(1, num + 1)]


def get_squares(num):
    """
    Get the first n triangular numbers.
    """
    return [int(i**2) for i in range(1, num + 1)]


def get_pentagonals(num):
    """
    Get the first n pentagonal numbers.
    """
    return [int(i * (3 * i - 1) / 2) for i in range(1, num + 1)]


def get_hexagonals(num):
    """
    Get the first n hexagonal numbers.
    """
    return [int(i * (2 * i - 1)) for i in range(1, num + 1)]


def get_heptagonals(num):
    """
    Get the first n heptagonal numbers.
    """
    return [int(i * (5 * i - 3) / 2) for i in range(1, num + 1)]


def get_octagonals(num):
    """
    Get the first n octagonal numbers.
    """
    return [int(i * (3 * i - 2)) for i in range(1, num + 1)]


class Modulos:
    """
    Basic computations in a modulos scope.
    This is equivalent to the Z_n group.
    """

    def __init__(self, mod):
        self.__mod = mod

    def identity(self, num):
        """
        The identity operator in Z_n.
        """
        return num % self.__mod

    def add(self, term_a, term_b):
        """
        The addition operator in Z_n.
        """
        return self.identity(term_a + term_b)

    def multiply(self, term_a, term_b):
        """
        The multiplication operator in Z_n.
        """
        return self.identity(self.identity(term_a) * self.identity(term_b))


class Combination:  # pylint: disable=too-few-public-methods
    """
    Calculates n-choose-k combinations.
    Uses a cache for repeated calcuation.
    """

    def __init__(self):
        self.cache = {}

    def n_choose_k(self, n, k):  # pylint: disable=invalid-name
        """
        Computes nCk, i.e. n-choose-k.
        """
        # sanity check
        assert isinstance(n, int) and n >= 1
        assert isinstance(k, int) and k >= 0
        # cache lookup
        if (n, k) in self.cache.keys():
            return self.cache[(n, k)]
        # base case: k = 0
        if k == 0:
            value = 1
        # symmetric case: k > n // 2
        elif k > n // 2:
            value = self.n_choose_k(n, n - k)
        # common case
        else:
            value = self.n_choose_k(n, k - 1) * (n - k + 1) / k
        # store result to cache
        self.cache[(n, k)] = int(value)
        return int(value)


class TexasHoldem:
    """
    Compares Poker hands according to the rules of Texas Holdem.
    """

    def __init__(self):
        self.value_mapping = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }

    def values_and_suites(self, cards):
        """
        Determine the values and suites of a 5-card hand.
        Example card: ('A', 'C') for A Clubs, ('T', 'H') for 10 Hearts.
        """
        from collections import defaultdict

        assert len(cards) == 5
        value_count = defaultdict(int)
        suite_count = defaultdict(int)
        for _value_raw, _suite in cards:
            _value = self.value_mapping[_value_raw]
            value_count[_value] += 1
            suite_count[_suite] += 1
        return value_count, suite_count

    def is_royal_flush(self, value_count, suite_count):
        """
        Check if a hand is a royal flush.
        """
        straight_flush = self.is_straight_flush(value_count, suite_count)
        if straight_flush:
            if straight_flush[1] == 14:
                return tuple([9] + list(straight_flush)[1:])
        return False

    def is_straight_flush(self, value_count, suite_count):
        """
        Check if a hand is a straight flush.
        """
        straight = self.is_straight(value_count, suite_count)
        flush = self.is_flush(value_count, suite_count)
        if straight and flush:
            return tuple([8] + list(straight)[1:])
        return False

    def is_four_of_a_kind(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a four of a kind.
        """
        if len(value_count.values()) == 2:
            if max(list(value_count.values())) == 4:
                for _key, _value in value_count.items():
                    if _value == 4:
                        quartuple = _key
                    else:
                        single = _key
                return (7, quartuple, single)
        return False

    def is_full_house(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a full house.
        """
        if len(value_count.values()) == 2:
            if max(list(value_count.values())) == 3:
                for _key, _value in value_count.items():
                    if _value == 3:
                        triple = _key
                    else:
                        double = _key
                return (6, triple, double)
        return False

    def is_flush(self, value_count, suite_count):  # pylint: disable=no-self-use
        """
        Check if a hand is a flush.
        """
        if len(suite_count.values()) == 1:
            if list(suite_count.values())[0] == 5:
                high_card = max(value_count.keys())
                return (5, high_card)
        return False

    def is_straight(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a straight.
        """
        if max(value_count.values()) == 1 and min(value_count.keys()) + 4 == max(
            value_count.keys()
        ):
            high_end = max(value_count.keys())
            return (4, high_end)
        return False

    def is_three_of_a_kind(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a three of a kind.
        """
        if len(value_count.values()) == 3:
            if max(list(value_count.values())) == 3:
                high_cards = []
                for _key, _value in value_count.items():
                    if _value == 3:
                        triple = _key
                    else:
                        high_cards.append(_key)
                high_cards = sorted(high_cards, reverse=True)
                return tuple([3, triple] + high_cards)
        return False

    def is_two_pairs(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a two pairs.
        """
        if len(value_count.values()) == 3:
            if max(list(value_count.values())) == 2:
                doubles = []
                for _key, _value in value_count.items():
                    if _value == 2:
                        doubles.append(_key)
                    else:
                        high_card = _key
                doubles = sorted(doubles, reverse=True)
                return tuple([2] + doubles + [high_card])
        return False

    def is_one_pair(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a one pair.
        """
        if len(value_count.values()) == 4:
            high_cards = []
            for _key, _value in value_count.items():
                if _value == 2:
                    double = _key
                else:
                    high_cards.append(_key)
            high_cards = sorted(high_cards, reverse=True)
            return tuple([1] + [double] + high_cards)
        return False

    def is_high_card(
        self, value_count, suite_count
    ):  # pylint: disable=unused-argument, no-self-use
        """
        Check if a hand is a high card.
        """
        if len(value_count.values()) == 5:
            high_cards = sorted(list(value_count.keys()), reverse=True)
            return tuple([0] + high_cards)
        return False

    def evaluate_hand(self, cards):
        """
        Determine the type and power of a hand.
        Example card: ('A', 'C') for A Clubs, ('10', 'H') for 10 Hearts.
        """
        value_count, suite_count = self.values_and_suites(cards)
        for _possibility in [
            self.is_royal_flush,
            self.is_straight_flush,
            self.is_four_of_a_kind,
            self.is_full_house,
            self.is_flush,
            self.is_straight,
            self.is_three_of_a_kind,
            self.is_two_pairs,
            self.is_one_pair,
            self.is_high_card,
        ]:
            matched = _possibility(value_count, suite_count)
            if matched:
                return matched
        raise ValueError("Expected at least one type of hand to be matched.")


class CliqueFinder:  # pylint: disable=too-few-public-methods
    """
    Given a graph, find the cliques in it.
    """

    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list[:]
        self.num_vertices = len(self.adjacency_list)
        self.cliques = {}
        self.compute(2)

    def compute(self, k):
        """
        Compute k-cliques in the graph.
        """
        from collections import defaultdict

        assert isinstance(k, int) and k >= 2
        # look-up case
        if k in self.cliques.keys():
            return self.cliques[k]

        k_cliques = set()
        # base case: k = 2
        if k == 2:
            for i in range(0, self.num_vertices):
                for j in self.adjacency_list[i]:
                    if i < j:
                        k_cliques.add((i, j))
        # common case: recursion
        else:
            # find all the k-1 cliques
            lower_cliques = self.compute(k - 1)
            for _clique in lower_cliques:
                _clique_set = set(_clique)
                # use a dict to find vertices that are connected to everyone in the clique
                degree = defaultdict(int)
                for i in _clique:
                    for j in self.adjacency_list[i]:
                        if j not in _clique_set:
                            degree[j] += 1
                for _key in degree.keys():
                    if degree[_key] == len(_clique):
                        new_clique = tuple(sorted(list(_clique) + [_key]))
                        k_cliques.add(new_clique)
        self.cliques[k] = k_cliques
        return k_cliques


def reverse_number(num):
    """
    Reverse a number.
    """
    return int(str(num)[::-1])


def xor_decipher(text, key):
    """
    Decipher a message using XOR.
    text -- a list of integers corresponding to the ASCII value of characters.
    key -- a list of characters used as keys.
    """
    deciphered = []
    key_length = len(key)
    key_ascii = [ord(_k) for _k in key]
    for i, _ascii in enumerate(text):
        deciphered.append(chr(_ascii ^ key_ascii[i % key_length]))
    return "".join(deciphered)


def max_sum_path_in_triangle(arr, row_idx=-1):
    """
    Given a triangle-shaped array, determine the max sum of elements along a downward path.
    arr -- the input array.
    row_idx -- the index of the row where the path terminates.
    Example:
       3
      7 4
     2 4 6
    8 5 9 3
    The max sum is 3 + 7 + 4 + 9 = 23.
    """
    # dynamic programming: tile it up by cumulative scores, row by row
    points = []
    for i, _row in enumerate(arr):
        # base case: the first row
        if i == 0:
            points.append(_row[:])
        else:
            tmp_row = []
            last_idx = len(_row) - 1
            for j, _num in enumerate(_row):
                # special case: the left-most element of a row
                if j == 0:
                    parent_value = points[i - 1][0]
                # special case: the right-most element of a row
                elif j == last_idx:
                    parent_value = points[i - 1][j - 1]
                # common case: a middle element of a row
                else:
                    parent_value = max(points[i - 1][j - 1], points[i - 1][j])
                tmp_row.append(parent_value + _row[j])
            points.append(tmp_row[:])
    return max(points[row_idx])


def continued_fraction_representation(num, max_terms=10**5, check_loop_length=30):
    """
    Warning: this implementation gets numerically unstable for long sequences.
    Given a positive Decimal b > 1, represent it as a sequence of integers {a_n}, such that
    b -> floor(b), a_1, a_2, ...
    1 / (b - floor(b)) -> a_1, a_2, a_3, ...
    Also detects if such a sequence has a loop.
    """
    from math import floor
    from decimal import Decimal

    assert isinstance(num, Decimal)
    int_part = floor(num)
    sequence = [int_part]
    residue = num - Decimal(int_part)
    reciprocal_monitor = dict()
    loop_start, loop_end = None, None
    while len(sequence) < max_terms and residue != 0.0:
        reciprocal = Decimal(1.0) / residue
        int_part = floor(reciprocal)
        sequence.append(int_part)
        residue = reciprocal - Decimal(int_part)
        # if the reciprocal has shown up before, we've found a loop in the sequence
        identifier = str(reciprocal)
        identifier = identifier[: min(check_loop_length, len(identifier))]
        if identifier in reciprocal_monitor:
            loop_start, loop_end = reciprocal_monitor[identifier], len(sequence) - 1
            break
        else:
            reciprocal_monitor[identifier] = len(sequence) - 1
    return sequence, loop_start, loop_end


def sqrt_continued_fraction_generator(num):
    """
    Takes the square root of a number, build a continued fraction sequence and put that into a generator
    """
    import sympy

    return sympy.ntheory.continued_fraction_iterator(sympy.sqrt(num))


def compile_continued_fraction_representation(seq):
    """
    Compile an integer sequence (continued fraction representation) into its corresponding fraction.
    """
    from fractions import Fraction

    # sanity check
    assert seq
    # initialize the value to be returned by working backwards from the last number
    retval = Fraction(1, seq.pop())
    # keep going backwords till the start of the sequence
    while seq:
        retval = 1 / (seq.pop() + retval)
    return retval


def solve_pells_equation(coeff_n):
    """
    Solver of Pell's equation, i.e. x^2 - n * y^2 = 1.
    Makes use of continued fraction representation of sqrt(n).
    Reference:
    https://en.wikipedia.org/wiki/Continued_fraction#Infinite_continued_fractions_and_convergents
    """
    assert isinstance(coeff_n, int) and coeff_n > 1
    from fractions import Fraction

    # get the continued fraction sequence
    sequence = sqrt_continued_fraction_generator(coeff_n)
    # keep a cache of the previous two terms
    cache = []
    # take advantage of that enumerate() and generators are both lazy
    for i, term in enumerate(sequence):
        _term = int(term)
        # base case: the first term
        if i == 0:
            _frac = Fraction(_term, 1)
        # base case: the second term
        elif i == 1:
            _frac = cache[-1] + Fraction(1, _term)
        # common case: recursion
        else:
            _numer = _term * cache[-1].numerator + cache[-2].numerator
            _denom = _term * cache[-1].denominator + cache[-2].denominator
            _frac = Fraction(_numer, _denom)
            cache.pop(0)
        cache.append(_frac)
        # check the fraction
        target = _frac.numerator**2 - coeff_n * _frac.denominator**2
        if target == 1:
            return (_frac.numerator, _frac.denominator)
    return (-1, -1)


def euler_totient(num, factors):
    """
    Given a number n and all its distinct prime factors, compute φ(n).
    The insight is that for every distinct prime factor p of a number n, the portion of numbers coprime to A "decays" by exactly (1/p).
    """
    from decimal import Decimal

    totient = Decimal(num)
    for _factor in factors:
        totient *= Decimal(_factor - 1) / Decimal(_factor)
    return int(totient)


def related_by_digit_permutation(num_a, num_b):
    """
    Check if two numbers are related by digit permutation.
    """
    from collections import Counter

    return Counter(str(num_a)) == Counter(str(num_b))


class LatticeGraph2D:
    """
    A 2-dimensional lattice where adjacent vertices may be connected.
    """

    def __init__(self, matrix, neighbor_function, weight_function):
        """
        Initialize the lattice by defining which vertices are connected without assuming the size of the lattice.
        neighbor_function(row_idx, col_idx, row_dim, col_dim) -- returns a list of (row_idx, col_idx) neighbors.
        weight_function(matrix, head_row_idx, head_col_idx, tail_row_idx, tail_col_idx) -- returns the weight of the edge from head to tail.
        """
        self.lattice = matrix
        self.row_dim = len(self.lattice)
        self.col_dim = len(self.lattice[0])
        self.neighbor_function = neighbor_function
        self.weight_function = weight_function
        self.consistency_check()
        self.build_adjacency_list()

    def consistency_check(self):
        """
        Check that
        (1) the lattice is indeed rectangular;
        (2) the neighbor function is callable;
        (3) the weight function is callable.
        """
        for _row in self.lattice:
            assert len(_row) == self.col_dim
        assert callable(self.neighbor_function)
        assert callable(self.weight_function)

    def flatten_index(self, i, j):
        """
        Flatten a 2D index to a 1D index.
        """
        return i * self.col_dim + j

    def unflatten_index(self, idx):
        """
        Unflatten a 1D index to a 2D index.
        """
        return idx // self.col_dim, idx % self.col_dim

    def build_adjacency_list(self):
        """
        Given a neighbor function and a weight function, build an adjacency list with edge weights.
        """
        # initialize adjacency list
        self.adjacency_list = []
        for i in range(self.row_dim):
            for j in range(self.col_dim):
                # get index for the current vertex and check consistency with the adjacency list
                head_index = self.flatten_index(i, j)
                assert len(self.adjacency_list) == head_index

                # build the contribution to the adjacency list from the current vertex
                connectivity = []
                neighbors = self.neighbor_function(i, j, self.row_dim, self.col_dim)
                for _neighbor_i, _neighbor_j in neighbors:
                    tail_index = self.flatten_index(_neighbor_i, _neighbor_j)
                    weight = self.weight_function(
                        self.lattice, i, j, _neighbor_i, _neighbor_j
                    )
                    connectivity.append((tail_index, weight))

                # update adjacency list
                self.adjacency_list.append(connectivity[:])

    def dijkstra_shortest_paths(self, i, j):
        """
        Find the shortest path from source (i, j).
        """
        distances, paths = dijkstra(self.adjacency_list, self.flatten_index(i, j))
        return distances, paths


def dijkstra(adjacency_dist_list, i):
    """
    Dijkstra's algorithm for shortest paths where edge lengths are non-negative.
    Args:
    adjacency_dist_list - adjacency list where adjacency_dist_list[i] is a list of (neighbor, distance) tuples.
    i - the source index to compute the distance from.
    """
    from datastruct import Heap

    # determine the number of nodes
    num_vertices = len(adjacency_dist_list)
    # initialize a list of distances
    distances = [-1] * num_vertices
    # initialize a list of shortest paths
    paths = [[]] * num_vertices
    # put the source node as a (distance, index) tuple in a heap
    heap = Heap([(0, i, [i])])
    # an iteration similar to breadth-first search
    while heap.values:
        dist_ij, j, path_ij = heap.extract()
        # if node j has not been visited before, update its distances and put its unvisited neighbors in the heap
        if distances[j] < 0:
            distances[j] = dist_ij
            paths[j] = path_ij
            for k, dist_jk in adjacency_dist_list[j]:
                assert dist_jk >= 0
                if distances[k] < 0:
                    heap.insert((dist_ij + dist_jk, k, path_ij + [k]))
    return distances, paths


class FloydWarshall:
    """
    Implementation of the Floyd-Warshall algorithm which computes all-pairs shortest distances and paths.
    Uses O(n^2) memory with optimized constant factor.
    Args:
    num_vertices - the number of vertices in the graph.
    edges - the edges in the graph, each being a (head, tail, weight) tuple.
    """

    def __init__(self, num_vertices, edges):
        """
        Initialize attributes of the following purpose:
        self.__min_distance - an array to hold shortest distances
        self.__max_internal - an array to hold max internal node indices in each path, which are used to restore paths
        self.__cap_internal - a parameter that controls the largest internal node that is permitted
        self.__num_vertices - the number of vertices in the graph
        self.__negative_cycle - whether the graph is known to contain a negative cycle
        Args:
        num_vertices - the number of vertices in the graph.
        edges - the edges in the graph, each being a (head, tail, weight) tuple.
        """
        import numpy

        self.__min_distance = numpy.full((num_vertices, num_vertices), numpy.inf)
        self.__max_internal = numpy.full((num_vertices, num_vertices), numpy.NAN)
        for i in range(0, num_vertices):
            # distances from a node to itself are zero
            self.__min_distance[i][i] = 0
            # an empty path has no internal node
            self.__max_internal[i][i] = -1
        # update distances for single-edge paths that don't contain internal nodes
        for _edge in edges:
            i, j, weight = _edge
            self.__min_distance[i][j] = weight
            # a single-edge path has no internal node
            self.__max_internal[i][j] = -1

        self.__cap_internal = -1
        self.__num_vertices = num_vertices
        self.__negative_cycle = False

    def __bump_cap_internal(self, verbose=True):
        """
        Allow one more node to be used as internal nodes in a path.
        Recompute the shortest distances and max internal nodes accordingly.
        """
        # if all nodes are already allowed or if a negative cycle has been detected, halt and return
        if self.__cap_internal >= self.__num_vertices - 1 or self.__negative_cycle:
            return
        if verbose:
            print(
                "Now running {0} out of {1} iterations.. ".format(
                    self.__cap_internal + 2, self.__num_vertices
                ),
                end="\r",
            )
        self.__cap_internal += 1
        self.__update_distances()

    def __update_distances(self):
        """
        Subroutine used in a single iteration to update all the pairwise shortest distances after bump_cap_internal() allows another internal node.
        """
        for i in range(0, self.__num_vertices):
            for j in range(0, self.__num_vertices):
                self.__update_single_pair(i, j)

    def __update_single_pair(self, i, j):
        """
        Subroutine to update the shortest disance, and the max-index internal node associated, from node i and node j.
        """
        update_value = (
            self.__min_distance[i][self.__cap_internal]
            + self.__min_distance[self.__cap_internal][j]
        )
        # distance updates can be done in-place because all values used to compute update_value sit in the union of a row and a column that never get themselves updated in this iteration. This saves one copy of self.__min_distance from memory.
        if update_value < self.__min_distance[i][j]:
            # there is a negative cycle if and only if node i has a negative path to itself
            if i == j:
                # sanity check that the cycle length is indeed negative; assertion error indicates a bug in the initialization of distances
                assert update_value < 0
                self.__negative_cycle = True
                return
            self.__min_distance[i][j] = update_value
            self.__max_internal[i][j] = self.__cap_internal

    def get_distances(self):
        """
        Launch the algorithm to compute shortest distances and paths.
        """
        for _i in range(self.__cap_internal, self.__num_vertices - 1):
            self.__bump_cap_internal()
        if self.__negative_cycle:
            raise ValueError("The graph contains a negative cycle.")
        return self.__min_distance

    def get_path(self, source, destination):
        """
        Backtrack and compute the shortest path from the source node to the destination node.
        If a path does exist, the running time is linear to the number of hops between source and destination.
        """
        import numpy

        # base case: destination unreachable from source
        if numpy.isinf(self.__min_distance[source][destination]):
            assert numpy.isnan(self.__max_internal[source][destination])
            return []
        # base case: destination is the same as source
        if source == destination:
            return [source]

        internal_node = int(self.__max_internal[source][destination])
        # base case: destination is one hop from source
        if internal_node < 0:
            return [source, destination]
        # common case: internal node found, start recursive call
        if internal_node >= 0:
            return self.get_path(source, internal_node)[:-1] + self.get_path(
                internal_node, destination
            )
        return ValueError("Expected one of the previous if statements to be true.")


def DFS_TS_subroutine(adjacency_list, explored, current_label, labels, s):
    """
    Non-recursive subroutine called by DFS_TS.
    Args:
    adjacency_list - the graph in its adjacency list representation.
    explored - whether each node has been explored.
    current_label - the next label to be assigned.
    labels - the list holding the labels of each node.
    s - stack to assist this subroutine.
    """
    # mark node i as explored
    i = s[-1]
    explored[i] = 1
    exhausted = True
    for j in adjacency_list[i]:
        # stack each neighbor that has not yet been explored
        if not bool(explored[j]):
            exhausted = False
            s.append(j)
    if exhausted:
        s.pop()
        # assign label and decrement it
        if not labels[i] >= 0:
            labels[i] = current_label
            current_label -= 1
    return current_label


def DFS_TS(adjacency_list):
    """
    Depth-first search of an adjacency list to compute a topological ordering.
    The topological labels will range from 0 to n-1.
    """
    # determine the number of nodes
    n = len(adjacency_list)
    # initialize explored statuses
    explored = [0] * n
    # initialize the topological ordering to be returned
    labels = [-1] * n
    # initialize the next label to be assigned
    current_label = n - 1
    # loop over all nodes
    for i in range(0, n):
        s = []
        # call the subroutine on nodes that have not yet been explored
        if not bool(explored[i]):
            s.append(i)
            while len(s) > 0:
                current_label = DFS_TS_subroutine(
                    adjacency_list, explored, current_label, labels, s
                )
    return labels


def reverse_adj_list(adjacency_list):
    """
    Reverse an adjacency list.
    This is only relevant for directed graphs. For undirected graphs, the reverse is just the same as the original.
    """
    # determine the number of vertices
    n = len(adjacency_list)
    # initialzie the adjacency list to be returned
    retlist = []
    for i in range(0, n):
        retlist.append([])
    # loop over all nodes
    for i, l in enumerate(adjacency_list):
        # create an edge from node j to node i
        for j in l:
            retlist[j].append(i)
    return retlist


def DFS_SCC_subroutine(adjacency_list, explored, comp, s):
    """
    Subroutine called by DFS_SCC() to determine strongly connected components from node i.
    Designed to be non-recursive so that large graphs don't crash the stack.
    """
    i = s.pop()
    if not bool(explored[i]):
        # mark node as explored
        explored[i] = 1
        # add node to the strongly connected component
        comp.append(i)
    # loop over neighbors and recursively call subroutine if not yet visited
    for j in adjacency_list[i]:
        if not bool(explored[j]):
            s.append(j)


def DFS_SCC(adjacency_list):
    """
    Compute the strongly connected components of a graph.
    This implements Kosaraju's two-pass DFS algorithm.
    Designed to be non-recursive so that large graphs don't crash the stack.
    """
    # determine the number of nodes
    n = len(adjacency_list)
    # compute the reversed graph
    adj_rev = reverse_adj_list(adjacency_list)
    # sort the nodes based on their label produced from the reversed graph
    rev_labels = list(zip(range(0, n), DFS_TS(adj_rev)))
    ordering = [r[0] for r in sorted(rev_labels, key=lambda x: x[1])]
    # initialize explored statuses
    explored = [0] * n
    # initialize the list of strongly connected components
    components = []
    # loop over all nodes in the specifically determined order
    for i in ordering:
        # if not already explored, compute the SCC associated with the node
        if not bool(explored[i]):
            comp = []
            s = [i]
            while len(s) > 0:
                DFS_SCC_subroutine(adjacency_list, explored, comp, s)
            # add SCC to the list
            components.append(comp)
    return components


def prim_mst(adj_list):
    """
    Prim's minimum spanning tree algorithm.
    Assumes that the graph is connected so a MST exists.
    """
    from datastruct import Heap

    minheap = Heap([])
    spanned = set()

    if len(adj_list) < 2:
        return []

    def span(i):
        if i in spanned:
            return False

        spanned.add(i)
        for j, _cost in adj_list[i]:
            if j in spanned:
                continue
            minheap.insert((_cost, i, j))
        return True

    mst = []
    span(0)
    while minheap.values:
        _cost, i, j = minheap.extract()
        if span(j):
            mst.append((_cost, i, j))
    assert (
        len(mst) == len(adj_list) - 1
    ), "Mininum spanning tree should contain edges 1 fewer than the number of nodes"
    return mst


@wrappy.memoize(cache_limit=100000)
def num_desc_seq_given_total_and_head(total, head):
    """
    Subproblem in dynamic programming.
    Count the number of descending sequences given a total and the head.
    Note that a one-term sequence is also considered a sequence.
    """
    if total < 1 or head < 1:
        return 0

    # base case: sequence has only one term
    if total == head:
        return 1

    # recursive case: sequence has more than one term
    # the second term cannot exceed the head; take advantage of transitivity
    num_seq = 0
    for _second in range(1, head + 1):
        num_seq += num_desc_seq_given_total_and_head(total - head, _second)

    return num_seq


@wrappy.memoize(cache_limit=100000)
def num_desc_prime_seq_given_total_and_head(total, head, list_of_primes, set_of_primes):
    """
    Subproblem in dynamic programming.
    Using a pre-computed list & set of primes, count the number of descending prime sequences given a total and the head.
    Note that a one-term sequence is also considered a sequence.
    """
    # sanity check
    assert head in set_of_primes, f"total: {total}, head: {head}"
    assert total >= head, f"total: {total}, head: {head}"

    # base case: sequence has only one term
    if total == head:
        return 1

    # recursive case: sequence has more than one term
    # the second term cannot exceed the head; take advantage of transitivity
    num_seq = 0
    for _second in list_of_primes:
        if _second > head or _second > total - head:
            break
        else:
            num_seq += num_desc_prime_seq_given_total_and_head(
                total - head, _second, list_of_primes, set_of_primes
            )

    return num_seq


def pythagorean_triplets(
    bound, ratio_lower_bound=0.0, ratio_upper_bound=1.0, coprime=True
):
    """
    Generates coprime Pythagorean triplets where the greatest of the triplet is under some bound, and the generating (n, m) pairs are also optionally bounded.
    """
    from math import sqrt, ceil, floor

    if coprime:
        fac = Factorizer(bound)
    bound_for_iteration = ceil(sqrt(bound))
    triplets = []
    # use the formula: (m^2 - n^2)^2 + (2mn)^2 = (m^2 + n^2)^2
    for _m in tqdm(range(2, bound_for_iteration)):
        _n_upper = min(_m, ceil(ratio_upper_bound * _m))
        _n_lower = max(1, floor(ratio_lower_bound * _m))
        for _n in range(_n_lower, _n_upper):
            # calculate Pythagorean triplet
            term_a = _m**2 - _n**2
            term_b = 2 * _m * _n
            term_c = _m**2 + _n**2

            if coprime:
                # skip triplets that are not coprime
                a_factors = fac.factorize(term_a)
                keep = True
                for _factor in a_factors:
                    if term_b % _factor == 0:
                        keep = False
                        break
                if not keep:
                    continue

            if term_c <= bound:
                _triplet = tuple(sorted([term_a, term_b, term_c]))
                triplets.append(_triplet)
    return triplets


def generate_combinations_from_element_wise_choices(list_choices):
    assert list_choices
    if len(list_choices) == 1:
        assert isinstance(list_choices[0], list), "Expected a list"
        for _ in list_choices[0]:
            yield [_]
    else:
        for _ele in list_choices[0]:
            for _seq in generate_combinations_from_element_wise_choices(
                list_choices[1:]
            ):
                yield [_ele, *_seq]


def generate_combinations_from_integer_range(elements=6, low=0, high=9):
    """
    Recursive approach to generate distinct integer combinations.
    Combinations are always in ascending order.
    """
    eff_high = high + 1 - elements
    if elements == 1:
        for _value in range(low, eff_high + 1):
            yield [_value]
    else:
        for _value in range(low, eff_high + 1):
            for _arr in generate_combinations_from_integer_range(
                elements=elements - 1,
                low=_value + 1,
                high=high,
            ):
                yield [_value, *_arr]


def generate_partitions_of_identical_elements(
    total, num_partitions, min_partition=0, max_partition=None
):
    """
    Recursive approach to generate partitions (counts) of identical elements.
    Optionally specify the least number of elements per partition.
    """
    # base case: just 1 partition
    assert num_partitions > 0, f"Invliad number of partitions: {num_partitions}"
    if num_partitions == 1:
        yield [total]
    else:
        # pre-assign the least count required
        if min_partition > 0:
            total -= num_partitions * min_partition
        assert total >= 0, "Total number is too low"

        # partition based on the 0-required version
        min_greatest = ((total - 1) // num_partitions) + 1
        max_greatest = total if max_partition is None else min(total, max_partition)
        for _kept in range(min_greatest, max_greatest + 1):
            for _arr in generate_partitions_of_identical_elements(
                total - _kept,
                num_partitions - 1,
                min_partition=0,
                max_partition=_kept,
            ):
                yield [_ + min_partition for _ in [_kept, *_arr]]


def generate_all_partitions_from_list(arr):
    """
    Recursive approach to generate array element partitions.
    """
    if len(arr) == 0:
        yield [], []
    else:
        for _l, _r in generate_all_partitions_from_list(arr[:-1]):
            yield [*_l, arr[-1]], _r
            yield _l, [*_r, arr[-1]]


class GrowingGeode:
    """
    "Geode" in 3D space that grows toward cubes next to its surface.
    """

    @staticmethod
    def slices_grown_from_unit(steps=0):
        slices = [1]
        if steps == 0:
            return slices
        for _ in range(1, steps + 1):
            slices.append(slices[-1] + _ * 4)
        return [*slices, *slices[-2::-1]]


class IntegerModulos:
    """
    Integer in a modulos space.
    """

    def __init__(self, value, modulos):
        assert isinstance(value, int)
        assert isinstance(modulos, int)
        self.__value = value
        self.__modulos = modulos
        self._reset()

    def _reset(self):
        self.__value = self.__value % self.__modulos

    def __add__(self, num):
        assert isinstance(num, int)
        retval = IntegerModulos(self.__value + num, self.__modulos)
        return retval

    def __sub__(self, num):
        assert isinstance(num, int)
        retval = IntegerModulos(self.__value - num, self.__modulos)
        return retval

    def __mul__(self, num):
        assert isinstance(num, int)
        retval = IntegerModulos(self.__value * num, self.__modulos)
        return retval

    def value(self):
        return self.__value

    def __repr__(self):
        return self.__value.__repr__()

    def __str__(self):
        return self.__value.__str__()


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
    solution.pop(tuple([]))
    return solution


def is_special_sum_set(set_as_sorted_tuple, verbose=False):
    """
    Check if a set, represented as a sorted typle, is a special sum set.
    Any set A is a special sum set iff for any non-empty disjoint subsets B and C
    - S(B) != S(C)
    - If B contains more elements than C then S(B) > S(C)
    """
    # basic set check: no duplicates
    if len(set_as_sorted_tuple) != len(set(set_as_sorted_tuple)):
        return False

    # compute all the sums of non-empty subsets
    # note that we do not need to check for disjointness because
    # any common elements between B and C has no effect on the
    # comparison between either S(B) vs. S(C) or len(B) vs. len(C)
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


def disjoint_subset_pairs_count_by_sizes(parent_size, nonempty=True):
    comb = Combination()
    pairs_by_sizes = {}
    for i in range(1 if nonempty else 0, (parent_size // 2) + 1):
        for j in range(i, parent_size - i + 1):
            _sizes = (i, j)
            _count = comb.n_choose_k(parent_size, i + j) * comb.n_choose_k(i + j, i)
            if i == j:
                # eliminate double-counting which happens when the pair has the same size
                assert _count % 2 == 0
                _count //= 2
            pairs_by_sizes[_sizes] = _count
    return pairs_by_sizes


def block_tiling_flexible_1d(m, n):
    """
    Compute the number of ways to tile n black blocks with red paint.
    Each group of red paint must cover at least m consecutive blocks.
    """
    end_in_red = [*[0 for _ in range(m - 1)], 1]
    end_in_black = [1 for _ in range(m)]

    for i in range(m, n):
        _reds = end_in_red[i - 1] + end_in_black[i - m]
        _blacks = end_in_red[i - 1] + end_in_black[i - 1]
        end_in_red.append(_reds)
        end_in_black.append(_blacks)

    return end_in_red, end_in_black


def block_tiling_fixed_1d(m, n):
    """
    Compute the number of ways to tile n black blocks with red paint.
    Each group of red paint must cover exactly m consecutive blocks.
    """
    end_in_red_end = [*[0 for _ in range(m - 1)], 1]
    end_in_black = [1 for _ in range(m)]

    for i in range(m, n):
        _reds = end_in_red_end[i - m] + end_in_black[i - m]
        _blacks = end_in_red_end[i - 1] + end_in_black[i - 1]
        end_in_red_end.append(_reds)
        end_in_black.append(_blacks)

    return end_in_red_end, end_in_black


def block_tiling_multifixed_1d(m_values, n):
    """
    Compute the number of ways to tile n black blocks with red paint.
    Each group of red paint must cover exactly m consecutive blocks where m has multiple choices.
    """
    m_values = sorted(m_values)
    m_min = m_values[0]
    end_in_red_end = [*[0 for _ in range(m_min - 1)], 1]
    end_in_black = [1 for _ in range(m_min)]

    for i in range(m_min, n):
        _reds, _blacks = 0, 0
        for _m in m_values:
            if i < _m - 1:
                pass
            elif i == _m - 1:
                _reds += 1
            else:
                _reds += end_in_red_end[i - _m] + end_in_black[i - _m]
        _blacks = end_in_red_end[i - 1] + end_in_black[i - 1]
        end_in_red_end.append(_reds)
        end_in_black.append(_blacks)

    return end_in_red_end, end_in_black


class BouncyNumberHelper:
    COMBINATION = Combination()

    @staticmethod
    def monotone_numbers_given_num_digits_and_first(num_digits, first, increasing=True):
        assert first != 0
        slots = 10 - first if increasing else first + 1
        picks = num_digits - 1
        return BouncyNumberHelper.COMBINATION.n_choose_k(slots + picks - 1, picks)

    @staticmethod
    def bouncy_numbers_given_num_digits_and_first(num_digits, first):
        assert first != 0
        total = 10 ** (num_digits - 1)
        increasing = BouncyNumberHelper.monotone_numbers_given_num_digits_and_first(
            num_digits, first, increasing=True
        )
        decreasing = BouncyNumberHelper.monotone_numbers_given_num_digits_and_first(
            num_digits, first, increasing=False
        )
        # exactly 1 double-counting between increasing and decreasing, which is "flat"
        bouncy = total - increasing - decreasing + 1
        return bouncy

    @staticmethod
    def is_bouncy(num):
        chars = list(str(num))
        if sorted(chars) == chars:
            return False
        if sorted(chars, reverse=True) == chars:
            return False
        return True
