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

def factorize_with_cache(num, cache):
    '''
    Assumes that the factorization of every number less than num has been stored in cache.
    cache -- a dict of dicts, eg. {2: {2: 1}, 4: {2: 2}, 6: {2: 1, 3: 1}}
    '''
    from collections import defaultdict
    if num < 2:
        return {}
    # case 0: num is already in cache
    if num in cache.keys():
        return cache[num]
    # case 1: num has a nontrivial divisor -- copy its factorization and bump
    trial = 1
    while trial < num:
        trial += 1
        factor = num / trial
        if factor in cache.keys():
            # remark: one could get away with storing just trial and factor, and then use a reconstruction method, to reduce memory usage from questionaly O(n logn) to O(n), where n is the largest number to be factorized.
            _factorization = defaultdict(int)
            _factorization.update(cache[factor])
            _factorization[trial] += 1
            cache[num] = _factorization
            return _factorization
    # case 2: num is a prime -- add new factorization to cache
    _factorization = {num: 1}
    cache[num] = _factorization 
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
    powers = list(factors.values())
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
        The cache must be of increasing order.
        '''
        from math import sqrt, ceil
        trial = 2
        for _p in cache:
            assert _p < num
            if _p > ceil(sqrt(num)):
                break
            if num % _p == 0:
                return False
        cache.append(num) 
        return True        

    # first use a list for keeping primes in increasing order
    cache_primes = []
    for num in range(2, n):
        is_prime_with_cache(num, cache_primes)
    # return a set for quick lookup
    return set(cache_primes)

def is_1_to_n_pandigital(num, n):
    '''
    Determine if a number is 1-to-n pandigital.
    '''
    from collections import defaultdict
    digit_count = defaultdict(int)
    list_form = list(str(num))
    for _digit in list_form:
        digit_count[_digit] += 1
    check_for_digits = list(map(str, list(range(1, n+1))))  
    # every digit in digit count must be between 1 and n and appear exactly once
    for _key in digit_count.keys():
        if not _key in check_for_digits:
            return False
        if digit_count[_key] != 1:
            return False
    # every digit from 1 to n must be in the digit count
    for _digit in check_for_digits:
        if not _digit in digit_count.keys():
            return False
    return True

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
