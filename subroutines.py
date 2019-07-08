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
