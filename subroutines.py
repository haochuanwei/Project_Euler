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
    if divisor < num:
        return True
    else:
        return False

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

class Factorizer(object):
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

class DigitwiseInteger(object):
    '''
    Integer that is represented by the value on each digit.
    '''
    def __init__(self, num):
        '''
        Initialize from a usual integer.
        '''
        from collections import defaultdict
        self.orderDict = defaultdict(int)
        _digit = 0
        _value = num
        if _value == 0:
            return {0: 0}
        while _value != 0:
            self.orderDict[_digit] = _value % 10
            _digit += 1
            _value = _value // 10

    def reconstruct(self):
        '''
        Reconstruct the original number.
        '''
        retval = 0
        for _digit, _value in self.orderDict.items():
            retval += _value * (10 ** _digit)
        return retval

    def __consistency_check(self):
        assert DigitwiseInteger(self.reconstruct()).reconstruct() == self.reconstruct()

    def correct_digits(self, orderDict):
        '''
        Promote digits with value greater than or equal to 10.
        '''
        from collections import defaultdict
        retDict = defaultdict(int)
        digits = sorted(orderDict.keys())
        # check digits from low to high
        for _digit in digits:
            _value = orderDict[_digit]
            # pickup value
            retDict[_digit] += _value
            # promote if appropriate
            if retDict[_digit] >= 10:
                retDict[_digit+1] += retDict[_digit] // 10
                retDict[_digit]    = retDict[_digit] % 10
        return retDict

    def multiply_by_constant(self, c, in_place=False, check_result=False):
        '''
        Subroutine to multiply a number by c.
        orderDict -- defaultdict of magnitude -> value mapping.
        Eg. {0: 1, 1: 3, 2: 6} stands for 1*10^1 + 3*10^1 + 6*10^2 = 631.
        '''
        from collections import defaultdict
        # perform calculation digit-wise
        retDict = defaultdict(int)
        for _key, _value in self.orderDict.items():
            multiplied = _value * c
            shift = 0
            while multiplied > 0 or shift == 0:
                retDict[_key+shift] += (multiplied % 10)
                multiplied = multiplied // 10
                shift += 1
        # promote digits that have value greater than 10
        retDict = self.correct_digits(retDict)
        if in_place:
            self.orderDict = defaultdict(int)
            self.orderDict.update(retDict)
            if check_result:
                self.__consistency_check()
        return retDict

def factorial(n):
    '''
    Factorial.
    '''
    assert isinstance(n, int) and n >= 0
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

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

def all_primes_under(n):
    '''
    Compute all the prime numbers below n.
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
    for num in range(2, n):
        is_prime_with_cache(num, cache_primes)
    return cache_primes[:]

def is_m_to_n_pandigital(num, m, n):
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
    for _d in range(m, n+1):
        target_count[str(_d)] = 1
    # compare two sets
    if digit_count == target_count:
        return True
    else:
        print(digit_count, target_count)
        return False

def two_sum(arr, num):
    '''
    The two-sum problem where the input array is already in set form.
    '''
    combinations = []
    assert isinstance(arr, set)
    for a in arr:
        b = num - a
        if b >= a and b in arr:
            combinations.append((a, b))
    return combinations

def is_triangular(num):
    '''
    Determine if a number is of the form (1/2)n(n+1).
    '''
    from math import floor, sqrt
    assert isinstance(num, int) and num > 0
    near_sqrt = floor(sqrt(2 * num))
    if int((1/2) * near_sqrt * (near_sqrt + 1)) == num:
        return True
    else:
        return False

def permutations_m_to_n_str(m, n):
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
        if len(perms) < 1:
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
    for _d in range(m, n+1):
        permutations = add(permutations, str(_d))
    return permutations

def get_triangulars(n):
    '''
    Get the first n triangular numbers.
    '''
    return [int(i * (i + 1) / 2) for i in range(1, n+1)]

def get_pentagonals(n):
    '''
    Get the first n pentagonal numbers.
    '''
    return [int(i * (3 * i - 1) / 2) for i in range(1, n+1)]

def get_hexagonals(n):
    '''
    Get the first n hexagonal numbers.
    '''
    return [int(i * (2 * i - 1)) for i in range(1, n+1)]

class Modulos(object):
    '''
    Basic computations in a modulos scope.
    This is equivalent to the Z_n group.
    '''
    def __init__(self, mod):
        self.__mod = mod
   
    def identity(self, num):
        return num % self.__mod

    def add(self, a, b):
        return self.identity(a + b)

    def multiply(self, a, b):
        return self.identity(self.identity(a) * self.identity(b))

class Combination(object):
    '''
    Calculates n-choose-k combinations.
    Uses a cache for repeated calcuation.
    '''
    def __init__(self):
        self.cache = {}

    def n_choose_k(self, n, k):
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

class TexasHoldem(object):
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

    def is_four_of_a_kind(self, value_count, suite_count):
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

    def is_full_house(self, value_count, suite_count):
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

    def is_flush(self, value_count, suite_count):
        '''
        Check if a hand is a flush.
        '''
        if len(suite_count.values()) == 1:
            if list(suite_count.values())[0] == 5:
                high_card = max(value_count.keys())
                return (5, high_card)
        return False

    def is_straight(self, value_count, suite_count):
        '''
        Check if a hand is a straight.
        '''
        if max(value_count.values()) == 1 and min(value_count.keys()) + 4 == max(value_count.keys()):
            high_end = max(value_count.keys())
            return (4, high_end)
        return False

    def is_three_of_a_kind(self, value_count, suite_count):
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

    def is_two_pairs(self, value_count, suite_count):
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

    def is_one_pair(self, value_count, suite_count):
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

    def is_high_card(self, value_count, suite_count):
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

class CliqueFinder(object):
    '''
    Given a graph, find the cliques in it.
    '''
    def __init__(self, adjacency_list):
        self.A = adjacency_list[:]
        self.n = len(self.A)
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
            for i in range(0, self.n):
                for j in self.A[i]:
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
                    for j in self.A[i]:
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
