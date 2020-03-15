'''
Subroutines that may get used repeatedly across different problems.
'''
# pylint: disable=line-too-long, bad-whitespace

def least_divisor(num, floor=2):
    '''
    Find the least divisor of a number, above some floor.
    '''
    assert num >= floor
    trial = floor
    while num % trial != 0:
        trial += 1
    return trial

def has_nontrivial_divisor(num):
    '''
    Determines if a number has a nontrivial divisor.
    '''
    divisor = least_divisor(num)
    return bool(divisor < num)

def is_a_palindrome(num):
    '''
    Determines if a number is a palindrome.
    '''
    assert isinstance(num, int)
    str_form = str(num)
    n_digits = len(str_form)
    for k in range(0, (n_digits+1)//2):
        if str_form[k] != str_form[-1-k]:
            return False
    return True

class Factorizer(): # pylint: disable=too-few-public-methods
    '''
    A factorizer that makes use of multiple data structures.
    This is intended to be efficient for repeated factorizations.
    '''
    def __init__(self, bound=10**6):
        '''
        Initialize a cache of factorizations and precompute primes.
        bound -- the bound of numbers that the factorizer is expected to deal with.
        '''
        self.bound = bound
        self.cache = {}
        self._update_primes()

    def _check_bound(self, num):
        if num > self.bound:
            print("{0} exceeded the expected bound {1}. Updating known prime numbers.".format(num, self.bound))
            self.bound = num * 2
            self._update_primes()

    def _update_primes(self):
        '''
        Update the list and set of primes, up to some bound.
        '''
        self.list_primes = all_primes_under(self.bound)
        self.set_primes  = set(self.list_primes)

    def _least_divisor(self, num):
        '''
        Find the least divisor of a number.
        '''
        self._check_bound(num)
        for _p in self.list_primes:
            if num % _p == 0:
                return _p
        raise ValueError("Unexpected behavior: {0} is not divisible by any number from {1}".format(num, self.list_primes))

    def factorize(self, num):
        '''
        Factorize a number.
        '''
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
        factor  = int(num / divisor)
        _factorization = defaultdict(int)
        _factorization.update(self.factorize(factor))
        _factorization[divisor] += 1
        self.cache[num] = _factorization
        return _factorization

def restore_from_factorization(factorization):
    '''
    Restore the original number from a factorization.
    '''
    retval = 1
    for _base, _power in factorization.items():
        retval *= (int(_base) ** int(_power))
    return retval

def get_num_divisors(factorization):
    '''
    Determine the number of different divisors given a factorization.
    '''
    from functools import reduce
    powers = list(factorization.values())
    num_divisors = reduce(lambda x, y: x * y, [_p + 1 for _p in powers])
    return num_divisors

def get_sum_proper_divisors(factorization):
    '''
    Determine the sum of proper divisors given a factorization.
    '''
    sum_divisors = 1
    original_number = 1
    for _base, _power in factorization.items():
        factors = [_base ** k for k in range(0, _power+1)]
        sum_divisors *= sum(factors)
        original_number *= (_base ** _power)
    return sum_divisors - original_number

class DigitwiseInteger():
    '''
    Integer that is represented by the value on each digit.
    '''
    def __init__(self, num):
        '''
        Initialize from a usual integer.
        '''
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
        '''
        Reconstruct the original number.
        '''
        retval = 0
        for _digit, _value in self.order_dict.items():
            retval += _value * (10 ** _digit)
        return retval

    def __consistency_check(self):
        assert DigitwiseInteger(self.reconstruct()).reconstruct() == self.reconstruct()

    def multiply_by_constant(self, multiplier, in_place=False, check_result=False):
        '''
        Subroutine to multiply a number by multiplier.
        Eg. {0: 1, 1: 3, 2: 6} stands for 1*10^1 + 3*10^1 + 6*10^2 = 631.
        '''
        from collections import defaultdict
        def correct_digits(order_dict):
            '''
            Promote digits with value greater than or equal to 10.
            '''
            ret_dict = defaultdict(int)
            digits = sorted(order_dict.keys())
            # check digits from low to high
            for _digit in digits:
                _value = order_dict[_digit]
                # pickup value
                ret_dict[_digit] += _value
                # promote if appropriate
                if ret_dict[_digit] >= 10:
                    ret_dict[_digit+1] += ret_dict[_digit] // 10
                    ret_dict[_digit]    = ret_dict[_digit] % 10
            return ret_dict

        # perform calculation digit-wise
        ret_dict = defaultdict(int)
        for _key, _value in self.order_dict.items():
            multiplied = _value * multiplier
            shift = 0
            while multiplied > 0 or shift == 0:
                ret_dict[_key+shift] += (multiplied % 10)
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
    '''
    Factorial.
    '''
    assert isinstance(num, int) and num >= 0
    if num == 0:
        return 1
    return num * factorial(num-1)

def is_prime_given_factorization(factorization):
    '''
    Given a factorization dict, determine if the original number is a prime.
    '''
    if len(factorization.keys()) == 1:
        if list(factorization.values())[0] == 1:
            return True
    return False

def is_prime_given_primes(num, primes):
    '''
    Determine if a number is prime, given an ascending list of prime numbers below its square root.
    '''
    from math import floor, sqrt
    assert primes[-1] >= floor(sqrt(num))
    for _p in primes:
        if _p > floor(sqrt(num)):
            break
        if num % _p == 0:
            return False
    return True

def all_primes_under(bound):
    '''
    Compute all the prime numbers below a bound.
    '''
    def is_prime_with_cache(num, cache):
        '''
        This is a subroutine for dynamic programming.
        Given a cache of primes below the square root of a number, determine if it is prime.
        The cache must be of ascending order.
        '''
        from math import sqrt, ceil
        for _p in cache:
            if _p > ceil(sqrt(num)):
                break
            if num % _p == 0:
                return False
        cache.append(num)
        return True

    # first use a list for keeping primes in ascending order
    cache_primes = []
    for candidate in range(2, bound):
        is_prime_with_cache(candidate, cache_primes)
    return cache_primes[:]

def is_m_to_n_pandigital(num, bound_m, bound_n):
    '''
    Determine if a number is m-to-n pandigital.
    '''
    digit_count = dict()
    list_form = list(str(num))
    for _digit in list_form:
        # return early if any digit shows up more than once
        if _digit in digit_count.keys():
            return False
        digit_count[_digit] = 1
    target_count = dict()
    for _d in range(bound_m, bound_n+1):
        target_count[str(_d)] = 1
    # compare two sets
    if digit_count == target_count:
        return True
    return False

def two_sum(arr, num):
    '''
    The two-sum problem where the input array is already in set form.
    '''
    combinations = []
    assert isinstance(arr, set)
    for term_a in arr:
        term_b = num - term_a
        if term_b >= term_a and term_b in arr:
            combinations.append((term_a, term_b))
    return combinations

def is_triangular(num):
    '''
    Determine if a number is of the form (1/2)n(n+1).
    '''
    from math import floor, sqrt
    assert isinstance(num, int) and num > 0
    near_sqrt = floor(sqrt(2 * num))
    return bool(int((1/2) * near_sqrt * (near_sqrt + 1)) == num)

def permutations_m_to_n_str(bound_m, bound_n):
    '''
    Get all permutations of digits between m and n, in string form.
    Example:
    permutations_m_to_n_str(1, 3) -> ['123', '132', '213', '231', '312', '321']
    '''
    def add(perms, new_digit):
        '''
        Add a digit to existing permutations.
        Assumes that all existing permutations have the same length.
        '''
        # base case: no permutation so far
        if not perms:
            return [new_digit]
        # common case
        perm_length = len(perms[0])
        retlist = []
        for _perm in perms:
            new_perms = [(_perm[:i] + new_digit + _perm[i:]) for i in range(0, perm_length)]
            new_perms.append(_perm + new_digit)
            retlist += new_perms
        return retlist
    permutations = []
    for _d in range(bound_m, bound_n+1):
        permutations = add(permutations, str(_d))
    return permutations

def get_triangulars(num):
    '''
    Get the first n triangular numbers.
    '''
    return [int(i * (i + 1) / 2) for i in range(1, num+1)]

def get_squares(num):
    '''
    Get the first n triangular numbers.
    '''
    return [int(i ** 2) for i in range(1, num+1)]

def get_pentagonals(num):
    '''
    Get the first n pentagonal numbers.
    '''
    return [int(i * (3 * i - 1) / 2) for i in range(1, num+1)]

def get_hexagonals(num):
    '''
    Get the first n hexagonal numbers.
    '''
    return [int(i * (2 * i - 1)) for i in range(1, num+1)]

def get_heptagonals(num):
    '''
    Get the first n heptagonal numbers.
    '''
    return [int(i * (5 * i - 3) / 2) for i in range(1, num+1)]

def get_octagonals(num):
    '''
    Get the first n octagonal numbers.
    '''
    return [int(i * (3 * i - 2)) for i in range(1, num+1)]

class Modulos():
    '''
    Basic computations in a modulos scope.
    This is equivalent to the Z_n group.
    '''
    def __init__(self, mod):
        self.__mod = mod

    def identity(self, num):
        '''
        The identity operator in Z_n.
        '''
        return num % self.__mod

    def add(self, term_a, term_b):
        '''
        The addition operator in Z_n.
        '''
        return self.identity(term_a + term_b)

    def multiply(self, term_a, term_b):
        '''
        The multiplication operator in Z_n.
        '''
        return self.identity(self.identity(term_a) * self.identity(term_b))

class Combination(): # pylint: disable=too-few-public-methods
    '''
    Calculates n-choose-k combinations.
    Uses a cache for repeated calcuation.
    '''
    def __init__(self):
        self.cache = {}

    def n_choose_k(self, n, k): # pylint: disable=invalid-name
        '''
        Computes nCk, i.e. n-choose-k.
        '''
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
            value = self.n_choose_k(n, n-k)
        # common case
        else:
            value = self.n_choose_k(n, k-1) * (n - k + 1) / k
        # store result to cache
        self.cache[(n, k)] = int(value)
        return int(value)

class TexasHoldem():
    '''
    Compares Poker hands according to the rules of Texas Holdem.
    '''
    def __init__(self):
        self.value_mapping = {
            '2':   2, '3':  3, '4':  4, '5':  5,
            '6':   6, '7':  7, '8':  8, '9':  9,
            'T':  10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
        }

    def values_and_suites(self, cards):
        '''
        Determine the values and suites of a 5-card hand.
        Example card: ('A', 'C') for A Clubs, ('T', 'H') for 10 Hearts.
        '''
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
        '''
        Check if a hand is a royal flush.
        '''
        straight_flush = self.is_straight_flush(value_count, suite_count)
        if straight_flush:
            if straight_flush[1] == 14:
                return tuple([9] + list(straight_flush)[1:])
        return False

    def is_straight_flush(self, value_count, suite_count):
        '''
        Check if a hand is a straight flush.
        '''
        straight = self.is_straight(value_count, suite_count)
        flush = self.is_flush(value_count, suite_count)
        if straight and flush:
            return tuple([8] + list(straight)[1:])
        return False

    def is_four_of_a_kind(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a four of a kind.
        '''
        if len(value_count.values()) == 2:
            if max(list(value_count.values())) == 4:
                for _key, _value in value_count.items():
                    if _value == 4:
                        quartuple = _key
                    else:
                        single = _key
                return (7, quartuple, single)
        return False

    def is_full_house(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a full house.
        '''
        if len(value_count.values()) == 2:
            if max(list(value_count.values())) == 3:
                for _key, _value in value_count.items():
                    if _value == 3:
                        triple = _key
                    else:
                        double = _key
                return (6, triple, double)
        return False

    def is_flush(self, value_count, suite_count): # pylint: disable=no-self-use
        '''
        Check if a hand is a flush.
        '''
        if len(suite_count.values()) == 1:
            if list(suite_count.values())[0] == 5:
                high_card = max(value_count.keys())
                return (5, high_card)
        return False

    def is_straight(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a straight.
        '''
        if max(value_count.values()) == 1 and min(value_count.keys()) + 4 == max(value_count.keys()):
            high_end = max(value_count.keys())
            return (4, high_end)
        return False

    def is_three_of_a_kind(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a three of a kind.
        '''
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

    def is_two_pairs(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a two pairs.
        '''
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

    def is_one_pair(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a one pair.
        '''
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

    def is_high_card(self, value_count, suite_count): # pylint: disable=unused-argument, no-self-use
        '''
        Check if a hand is a high card.
        '''
        if len(value_count.values()) == 5:
            high_cards = sorted(list(value_count.keys()), reverse=True)
            return tuple([0] + high_cards)
        return False

    def evaluate_hand(self, cards):
        '''
        Determine the type and power of a hand.
        Example card: ('A', 'C') for A Clubs, ('10', 'H') for 10 Hearts.
        '''
        value_count, suite_count = self.values_and_suites(cards)
        for _possibility in [self.is_royal_flush, self.is_straight_flush, self.is_four_of_a_kind, self.is_full_house, self.is_flush, self.is_straight, self.is_three_of_a_kind, self.is_two_pairs, self.is_one_pair, self.is_high_card]:
            matched = _possibility(value_count, suite_count)
            if matched:
                return matched
        raise ValueError("Expected at least one type of hand to be matched.")

class CliqueFinder(): # pylint: disable=too-few-public-methods
    '''
    Given a graph, find the cliques in it.
    '''
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list[:]
        self.num_vertices = len(self.adjacency_list)
        self.cliques = {}
        self.compute(2)

    def compute(self, k):
        '''
        Compute k-cliques in the graph.
        '''
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
            lower_cliques = self.compute(k-1)
            for _clique in lower_cliques:
                _clique_set = set(_clique)
                # use a dict to find vertices that are connected to everyone in the clique
                degree = defaultdict(int)
                for i in _clique:
                    for j in self.adjacency_list[i]:
                        if not j in _clique_set:
                            degree[j] += 1
                for _key in degree.keys():
                    if degree[_key] == len(_clique):
                        new_clique = tuple(sorted(list(_clique) + [_key]))
                        k_cliques.add(new_clique)
        self.cliques[k] = k_cliques
        return k_cliques

def reverse_number(num):
    '''
    Reverse a number.
    '''
    return int(str(num)[::-1])

def xor_decipher(text, key):
    '''
    Decipher a message using XOR.
    text -- a list of integers corresponding to the ASCII value of characters.
    key -- a list of characters used as keys.
    '''
    deciphered = []
    key_length = len(key)
    key_ascii  = [ord(_k) for _k in key]
    for i, _ascii in enumerate(text):
        deciphered.append(chr(_ascii ^ key_ascii[i % key_length]))
    return ''.join(deciphered)

def max_sum_path_in_triangle(arr, row_idx=-1):
    '''
    Given a triangle-shaped array, determine the max sum of elements along a downward path.
    arr -- the input array.
    row_idx -- the index of the row where the path terminates.
    Example:
       3
      7 4
     2 4 6
    8 5 9 3
    The max sum is 3 + 7 + 4 + 9 = 23.
    '''
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
                    parent_value = points[i-1][0]
                # special case: the right-most element of a row
                elif j == last_idx:
                    parent_value = points[i-1][j-1]
                # common case: a middle element of a row
                else:
                    parent_value = max(points[i-1][j-1], points[i-1][j])
                tmp_row.append(parent_value + _row[j])
            points.append(tmp_row[:])
    return max(points[row_idx])

def continued_fraction_representation(num, max_terms=10**5, check_loop_length=30):
    '''
    Warning: this implementation gets numerically unstable for long sequences.
    Given a positive Decimal b > 1, represent it as a sequence of integers {a_n}, such that
    b -> floor(b), a_1, a_2, ...
    1 / (b - floor(b)) -> a_1, a_2, a_3, ...
    Also detects if such a sequence has a loop.
    '''
    from math import floor
    from decimal import Decimal
    assert isinstance(num, Decimal)
    int_part = floor(num)
    sequence = [int_part]
    residue  = num - Decimal(int_part)
    reciprocal_monitor = dict()
    loop_start, loop_end = None, None
    while len(sequence) < max_terms and residue != 0.0:
        reciprocal = Decimal(1.0) / residue
        int_part = floor(reciprocal)
        sequence.append(int_part)
        residue = reciprocal - Decimal(int_part)
        # if the reciprocal has shown up before, we've found a loop in the sequence
        identifier = str(reciprocal)
        identifier = identifier[:min(check_loop_length, len(identifier))]
        if identifier in reciprocal_monitor:
            loop_start, loop_end = reciprocal_monitor[identifier], len(sequence) - 1
            break
        else:
            reciprocal_monitor[identifier] = len(sequence) - 1
    return sequence, loop_start, loop_end

def sqrt_continued_fraction_generator(num):
    '''
    Takes the square root of a number, build a continued fraction sequence and put that into a generator
    '''
    import sympy
    return sympy.ntheory.continued_fraction_iterator(sympy.sqrt(num))

def compile_continued_fraction_representation(seq):
    '''
    Compile an integer sequence (continued fraction representation) into its corresponding fraction.
    '''
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
    '''
    Solver of Pell's equation, i.e. x^2 - n * y^2 = 1.
    Makes use of continued fraction representation of sqrt(n).
    Reference:
    https://en.wikipedia.org/wiki/Continued_fraction#Infinite_continued_fractions_and_convergents
    '''
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
            _frac  = Fraction(_numer, _denom)
            cache.pop(0)
        cache.append(_frac)
        # check the fraction
        target = _frac.numerator ** 2 - coeff_n * _frac.denominator ** 2
        if target == 1:
            return (_frac.numerator, _frac.denominator)
    return (-1, -1)

def euler_totient(num, factors):
    '''
    Given a number n and all its distinct prime factors, compute φ(n).
    The insight is that for every distinct prime factor p of a number n, the portion of numbers coprime to A "decays" by exactly (1/p).
    '''
    from decimal import Decimal
    totient = Decimal(num)
    for _factor in factors:
        totient *= Decimal(_factor - 1) / Decimal(_factor)
    return int(totient)

def related_by_digit_permutation(num_a, num_b):
    '''
    Check if two numbers are related by digit permutation.
    '''
    from collections import Counter
    return Counter(str(num_a)) == Counter(str(num_b))

class LatticeGraph2D():
    '''
    A 2-dimensional lattice where adjacent vertices may be connected.
    '''
    def __init__(self, matrix, neighbor_function, weight_function):
        '''
        Initialize the lattice by defining which vertices are connected without assuming the size of the lattice.
        neighbor_function(row_idx, col_idx, row_dim, col_dim) -- returns a list of (row_idx, col_idx) neighbors.
        weight_function(matrix, head_row_idx, head_col_idx, tail_row_idx, tail_col_idx) -- returns the weight of the edge from head to tail.
        '''
        self.lattice = matrix
        self.row_dim = len(self.lattice)
        self.col_dim = len(self.lattice[0])
        self.neighbor_function = neighbor_function
        self.weight_function   = weight_function
        self.consistency_check()
        self.build_adjacency_list()

    def consistency_check(self):
        '''
        Check that
        (1) the lattice is indeed rectangular;
        (2) the neighbor function is callable;
        (3) the weight function is callable.
        '''
        for _row in self.lattice:
            assert len(_row) == self.col_dim
        assert callable(self.neighbor_function)
        assert callable(self.weight_function)

    def flatten_index(self, i, j):
        '''
        Flatten a 2D index to a 1D index.
        '''
        return i * self.col_dim + j

    def unflatten_index(self, idx):
        '''
        Unflatten a 1D index to a 2D index.
        '''
        return idx // self.col_dim, idx % self.col_dim

    def build_adjacency_list(self):
        '''
        Given a neighbor function and a weight function, build an adjacency list with edge weights.
        '''
        # initialize adjacency list
        self.adjacency_list = []
        for i in range(self.row_dim):
            for j in range(self.col_dim):
                # get index for the current vertex and check consistency with the adjacency list
                head_index   = self.flatten_index(i, j)
                assert len(self.adjacency_list) == head_index

                # build the contribution to the adjacency list from the current vertex
                connectivity = []
                neighbors    = self.neighbor_function(i, j, self.row_dim, self.col_dim)
                for _neighbor_i, _neighbor_j in neighbors:
                    tail_index = self.flatten_index(_neighbor_i, _neighbor_j)
                    weight     = self.weight_function(self.lattice, i, j, _neighbor_i, _neighbor_j)
                    connectivity.append((tail_index, weight))

                # update adjacency list
                self.adjacency_list.append(connectivity[:])

    def dijkstra_shortest_paths(self, i, j):
        '''
        Find the shortest path from source (i, j).
        '''
        distances, paths = dijkstra(self.adjacency_list, self.flatten_index(i, j))
        return distances, paths

def dijkstra(adjacency_dist_list, i):
    '''
    Dijkstra's algorithm for shortest paths where edge lengths are non-negative.
    Args:
    adjacency_dist_list - adjacency list where adjacency_dist_list[i] is a list of (neighbor, distance) tuples.
    i - the source index to compute the distance from.
    '''
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
                    heap.insert((dist_ij+dist_jk, k, path_ij + [k]))
    return distances, paths

class FloydWarshall():
    '''
    Implementation of the Floyd-Warshall algorithm which computes all-pairs shortest distances and paths.
    Uses O(n^2) memory with optimized constant factor.
    Args:
    num_vertices - the number of vertices in the graph.
    edges - the edges in the graph, each being a (head, tail, weight) tuple.
    '''
    def __init__(self, num_vertices, edges):
        '''
        Initialize attributes of the following purpose:
        self.__min_distance - an array to hold shortest distances
        self.__max_internal - an array to hold max internal node indices in each path, which are used to restore paths
        self.__cap_internal - a parameter that controls the largest internal node that is permitted
        self.__num_vertices - the number of vertices in the graph
        self.__negative_cycle - whether the graph is known to contain a negative cycle
        Args:
        num_vertices - the number of vertices in the graph.
        edges - the edges in the graph, each being a (head, tail, weight) tuple.
        '''
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
        '''
        Allow one more node to be used as internal nodes in a path.
        Recompute the shortest distances and max internal nodes accordingly.
        '''
        # if all nodes are already allowed or if a negative cycle has been detected, halt and return
        if self.__cap_internal >= self.__num_vertices - 1 or self.__negative_cycle:
            return
        if verbose:
            print("Now running {0} out of {1} iterations.. ".format(self.__cap_internal+2, self.__num_vertices), end="\r")
        self.__cap_internal += 1
        self.__update_distances()

    def __update_distances(self):
        '''
        Subroutine used in a single iteration to update all the pairwise shortest distances after bump_cap_internal() allows another internal node.
        '''
        for i in range(0, self.__num_vertices):
            for j in range(0, self.__num_vertices):
                self.__update_single_pair(i, j)

    def __update_single_pair(self, i, j):
        '''
        Subroutine to update the shortest disance, and the max-index internal node associated, from node i and node j.
        '''
        update_value = self.__min_distance[i][self.__cap_internal] + self.__min_distance[self.__cap_internal][j]
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
        '''
        Launch the algorithm to compute shortest distances and paths.
        '''
        for _i in range(self.__cap_internal, self.__num_vertices-1):
            self.__bump_cap_internal()
        if self.__negative_cycle:
            raise ValueError("The graph contains a negative cycle.")
        return self.__min_distance

    def get_path(self, source, destination):
        '''
        Backtrack and compute the shortest path from the source node to the destination node.
        If a path does exist, the running time is linear to the number of hops between source and destination.
        '''
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
            return self.get_path(source, internal_node)[:-1] + self.get_path(internal_node, destination)
        return ValueError("Expected one of the previous if statements to be true.")
