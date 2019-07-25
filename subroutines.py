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

def multiply_by_constant(orderDict, c):
    '''
    Subroutine to multiply a number by c.
    orderDict -- defaultdict of magnitude -> value mapping.
    Eg. {0: 1, 1: 3, 2: 6} stands for 1*10^1 + 3*10^1 + 6*10^2 = 631.
    '''
    from collections import defaultdict
    def correct_digits(orderDict):
        '''
        Promote digits with value greater than or equal to 10.
        '''
        retDict = defaultdict(int)
        for _key, _value in orderDict.items():
            retDict[_key] += _value % 10
            if _value >= 10:
                retDict[_key+1] += _value // 10
        return retDict

    retDict = defaultdict(int)
    for _key, _value in orderDict.items():
        multiplied = _value * c
        shift = 0
        while multiplied > 0 or shift == 0:
            retDict[_key+shift] += (multiplied % 10)
            multiplied = multiplied // 10
            shift += 1
    return correct_digits(retDict)

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

def all_primes_under(n):
    '''
    Compute all the prime numbers below n.
    '''
    def is_prime_with_cache(num, cache):
        '''
        This is a subroutine for dynamic programming.
        Given a cache of primes below a number, determine if it is prime.
        The cache must be of ascending order.
        '''
        from math import sqrt, ceil
        for _p in cache:
            assert _p < num
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

