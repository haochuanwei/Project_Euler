def Euler_Problem_1(n=1000):
    '''
    If we list all the natural numbers below 10 that are multiples of 3 or 5, we get 3, 5, 6 and 9. The sum of these multiples is 23.
    Find the sum of all the multiples of 3 or 5 below 1000.
    '''
    # compute the number of such multiples in three sets
    num_multiples_of_3   = (n-1) // 3
    num_multiples_of_5   = (n-1) // 5
    num_shared_multiples = (n-1) // (3 * 5)

    # compute the sum of each set
    def sum_one_up_to_k(k):
        return (k ** 2 + k) / 2
    sum_multiples_of_3   = sum_one_up_to_k(num_multiples_of_3) * 3
    sum_multiples_of_5   = sum_one_up_to_k(num_multiples_of_5) * 5
    sum_shared_multiples = sum_one_up_to_k(num_shared_multiples) * (3 * 5)

    # use inclusion/exclusion to compute the final answer
    retval = sum_multiples_of_3 + sum_multiples_of_5 - sum_shared_multiples
    return retval

def Euler_Problem_2(n=4000000):
    '''
    Each new term in the Fibonacci sequence is generated by adding the previous two terms. By starting with 1 and 2, the first 10 terms will be:
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
    By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.
    '''
    # note that for every triple starting with (1, 2, 3), the middle value is even
    def new_triple(old_triple):
        _left, _mid, _right = old_triple
        left = _mid + _right
        mid = left + _right
        right = left + mid
        return (left, mid, right)

    retval = 0
    current_triple = (1, 2, 3)
    while True:
        if current_triple[1] > n:
            break
        retval += current_triple[1]
        current_triple = new_triple(current_triple)
    return retval

def Euler_Problem_3(n=600851475143):
    '''
    The prime factors of 13195 are 5, 7, 13 and 29.
    What is the largest prime factor of the number 600851475143 ?
    '''
    # start from 2 and keep dividing
    def least_divisor(num, floor=2):
        '''
        Find the least divisor of a number, above some floor.
        '''
        assert num >= floor
        trial = floor
        while num % trial != 0:
            trial += 1
        return trial

    greatest_divisor = 2
    value            = n
    while greatest_divisor < value:
        greatest_divisor = least_divisor(value, greatest_divisor)
        value           /= greatest_divisor

    return greatest_divisor

def Euler_Problem_4(n=3):
    '''
    A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 * 99.
    Find the largest palindrome made from the product of two 3-digit numbers.
    '''
    def is_a_palindrome(num):
        assert isinstance(num, int)
        str_form = str(num)
        n_digits = len(str_form)
        for k in range(0, (n_digits+1)//2):
            if str_form[k] != str_form[-1-k]:
                return False
        return True

    assert n >= 2
    greatest_product = 0
    # brute-force approach that searches the high end of possible products
    upper_bound = 10 ** n - 1
    lower_bound = int(0.9 * upper_bound)
    for p in range(upper_bound, lower_bound, -1):
        for q in range(upper_bound, lower_bound, -1):
            candidate = p * q
            if candidate > greatest_product:
                if is_a_palindrome(candidate):
                    greatest_product = candidate
    return greatest_product

def Euler_Problem_5(n=20):
    '''
    2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.
    What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?
    '''
    def lowest_divisor(num):
        assert num >= 2
        trial = 2
        while num % trial != 0:
            trial += 1
        return trial

    def subproblem(num, cache):
        '''
        Uses a cache to store intermediate results.
        '''
        assert isinstance(num, int)
        assert num > 0 
        # base case
        if num == 1:
            cache[num] = 1
            return cache[num] 

        # recursion / dynamic programming
        if not num-1 in cache.keys():
            prev = subproblem(num-1, cache)
        if prev % num == 0:
            cache[num] = prev
        else:
            factor = lowest_divisor(num)
            cache[num] = prev * factor
        return cache[num]

    solution_cache = {}
    return subproblem(n, solution_cache)

def Euler_Problem_6(n=100):
    '''
    The sum of the squares of the first ten natural numbers is,
    1^2 + 2^2 + ... + 10^2 = 385
    The square of the sum of the first ten natural numbers is,
    (1 + 2 + ... + 10)^2 = 55^2 = 3025
    Hence the difference between the sum of the squares of the first ten natural numbers and the square of the sum is 3025 - 385 = 2640.
    Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum.
    '''
    # looks like brute force gives you O(n) or O(n logn), which is not bad...
    # but we can do better with mathematical insight.
    def sum_of_integer_squares(k):
        '''
        Use the formula 1^2 + 2^2 + ... + n^2 = (n * (n+1) * (2n+1)) / 6.
        '''
        return (k * (k+1) * (2*k + 1)) / 6
   
    def square_of_integer_sums(k):
        '''
        Use the formula 1 + 2 + ... + n = n (n+1) / 2.
        '''
        return (n * (n+1) / 2) ** 2

    # O(logn) basic operations
    sqsum = square_of_integer_sums(n)
    sumsq = sum_of_integer_squares(n)
    return int(sqsum - sumsq)

def Euler_Problem_7(n=10001):
    '''
    By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
    What is the 10001st prime number?
    '''
    def has_nontrivial_divisor(num):
        assert num >= 2
        trial = 2
        while num % trial != 0:
            trial += 1
        if trial < num:
            return True
        else:
            return False
    
    # brute force: check numbers one by one
    value = 1
    num_primes = 0
    while num_primes < n:
        value += 1
        if not has_nontrivial_divisor(value):
            num_primes += 1
    return value

def Euler_Problem_8(n=13):
    '''
    The four adjacent digits in the 1000-digit number that have the greatest product are 9 * 9 * 8 * 9 = 5832.
    73167176531330624919225119674426574742355349194934
    96983520312774506326239578318016984801869478851843
    85861560789112949495459501737958331952853208805511
    12540698747158523863050715693290963295227443043557
    66896648950445244523161731856403098711121722383113
    62229893423380308135336276614282806444486645238749
    30358907296290491560440772390713810515859307960866
    70172427121883998797908792274921901699720888093776
    65727333001053367881220235421809751254540594752243
    52584907711670556013604839586446706324415722155397
    53697817977846174064955149290862569321978468622482
    83972241375657056057490261407972968652414535100474
    82166370484403199890008895243450658541227588666881
    16427171479924442928230863465674813919123162824586
    17866458359124566529476545682848912883142607690042
    24219022671055626321111109370544217506941658960408
    07198403850962455444362981230987879927244284909188
    84580156166097919133875499200524063689912560717606
    05886116467109405077541002256983155200055935729725
    71636269561882670428252483600823257530420752963450
    Find the thirteen adjacent digits in the 1000-digit number that have the greatest product. What is the value of this product?
    '''
    from functools import reduce
    # preprocessing
    series = '''\
    73167176531330624919225119674426574742355349194934\
    96983520312774506326239578318016984801869478851843\
    85861560789112949495459501737958331952853208805511\
    12540698747158523863050715693290963295227443043557\
    66896648950445244523161731856403098711121722383113\
    62229893423380308135336276614282806444486645238749\
    30358907296290491560440772390713810515859307960866\
    70172427121883998797908792274921901699720888093776\
    65727333001053367881220235421809751254540594752243\
    52584907711670556013604839586446706324415722155397\
    53697817977846174064955149290862569321978468622482\
    83972241375657056057490261407972968652414535100474\
    82166370484403199890008895243450658541227588666881\
    16427171479924442928230863465674813919123162824586\
    17866458359124566529476545682848912883142607690042\
    24219022671055626321111109370544217506941658960408\
    07198403850962455444362981230987879927244284909188\
    84580156166097919133875499200524063689912560717606\
    05886116467109405077541002256983155200055935729725\
    71636269561882670428252483600823257530420752963450\
    '''.replace(' ', '')
    series = list(map(int, list(series)))
    assert len(series) > n

    # linear scan of all products
    tmp_prod = reduce(lambda a, b: a * b, series[0:n])
    max_prod = tmp_prod
    # attempt to cheat the product at O(1)
    for k in range(1, len(series)-n):
        # attempt works if the term to be removed is nonzero
        if series[k-1] > 0:
            tmp_prod = tmp_prod * series[k+n-1] / series[k-1]
        # attmpt fails is the term to be removed is zero
        else:
            tmp_prod = reduce(lambda a, b: a * b, series[k:(k+n)])
        # update product
        if tmp_prod > max_prod:
            max_prod = tmp_prod
    return max_prod

def Euler_Problem_9(n=1000):
    '''
    A Pythagorean triplet is a set of three natural numbers, a < b < c, for which,
    a^2 + b^2 = c^2
    For example, 3^2 + 4^2 = 9 + 16 = 25 = 5^2.
    There exists exactly one Pythagorean triplet for which a + b + c = 1000.
    Find the product abc.
    '''
    import math
    assert n > 10
    # first assume that a <= b < c < n/2. Then, for c to be an integer we can't have a=b.
    # hence assume that a < b < c and that n/3 < 3.

    # brute-force O(n^2) approach
    for c in range(n//2, n//3, -1):
        c_sq = c ** 2
        for b in range(c-1, int(c / math.sqrt(2)), -1):
            a = n - c - b
            if a ** 2 + b ** 2 == c_sq:
                return a * b * c
    return -1

def Euler_Problem_10(n=2000000):
    '''
    The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.
    Find the sum of all the primes below two million.
    '''
    def divisible_by_prime(num, primes):
        assert num >= 2
        for _p in primes:
            if num % _p == 0:
                return True
        return False
    
    # brute force: check numbers one by one, starting at 3
    # also maintain a list of primes to divide by
    value = 2
    current_primes = [2]
    while value < n-1:
        if value % 10000 == 0:
            print(value)
        value += 1
        if not divisible_by_prime(value, current_primes):
            current_primes.append(value)
    return sum(current_primes)

def Euler_Problem_11(k=4):
    '''
    In the 20 * 20 grid below, four numbers along a diagonal line have been marked in red.
    08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
    49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
    81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
    52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
    22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
    24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
    32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
    67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
    24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
    21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
    78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
    16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
    86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
    19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
    04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
    88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
    04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
    20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
    20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
    01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48
    The product of these numbers is 26 * 63 * 78 * 14 = 1788696.
    What is the greatest product of four adjacent numbers in the same direction (up, down, left, right, or diagonally) in the 20 * 20 grid?
    '''
    # This problem always takes at least O(M^2) where M is the size of the grid.
    # Here's the argument:
    # Consider the reduced problem where we only care about adjacent numbers in the horizontal direction. Then we need a linear scan in each row, giving a running time of O(M^2).
    # Note the brute force takes O(M^2), which is good enough in asymtotic scaling.

    # construct the array
    import numpy as np
    arr_in_str_form = '''
    08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
    49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
    81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
    52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
    22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
    24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
    32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
    67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
    24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
    21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
    78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
    16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
    86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
    19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
    04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
    88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
    04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
    20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
    20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
    01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48
    '''
    arr_in_str_form = [_x for _x in arr_in_str_form.split('\n') if len(_x) > 0]
    arr = [[int(_z) for _z in _y.split(' ') if len(_z) > 0] for _y in arr_in_str_form]
    arr = np.array([_a for _a in arr if len(_a) > 0])
    assert arr.shape == (20, 20)
    
    # set up max product variable
    max_prod = 0

    # consider horizontal and vertical
    for i in range(0, 20):
        for j in range(0, 20-k+1):
            # horizontal
            value = 1
            for shift in range(0, k):
                value *= arr[i][j+shift]
            if value > max_prod:
                max_prod = value

            # vertical
            value = 1
            for shift in range(0, k):
                value *= arr[j+shift][i]
            if value > max_prod:
                max_prod = value

    # consider diagonal directions 
    for i in range(0, 20-k+1):
        for j in range(0, 20-k+1):
            # upper-left-to-lower-right
            value = 1
            for shift in range(0, k):
                value *= arr[i+shift][j+shift]
            if value > max_prod:
                max_prod = value

            # upper-right-to-lower-left
            value = 1
            for shift in range(0, k):
                value *= arr[i+shift][20-1-j-shift]
            if value > max_prod:
                max_prod = value

    return max_prod

def Euler_Problem_12(n=500):
    '''
    The sequence of triangle numbers is generated by adding the natural numbers. So the 7th triangle number would be 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28. The first ten terms would be:
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...
    Let us list the factors of the first seven triangle numbers:
     1: 1
     3: 1,3
     6: 1,2,3,6
    10: 1,2,5,10
    15: 1,3,5,15
    21: 1,3,7,21
    28: 1,2,4,7,14,28
    We can see that 28 is the first triangle number to have over five divisors.
    What is the value of the first triangle number to have over five hundred divisors?
    '''

    # subroutine to factorize with dynamic programming
    def factorize(num, cache):
        '''
        Assumes that the factorization of every number less than num has been stored in cache.
        cache -- a dict of dicts, eg. {2: {2: 1}, 4: {2: 2}, 6: {2: 1, 3: 1}}
        '''
        from collections import defaultdict
        trial = 1
        # case 1: num has a nontrivial divisor -- copy its factorization and bump
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

    def get_num_divisors(factorization):
        '''
        Determine the number of different divisors given a factorization.
        '''
        from functools import reduce
        powers = list(factors.values())
        num_divisors = reduce(lambda x, y: x * y, [_p + 1 for _p in powers])
        return num_divisors

    # initialize cache
    cache_factorizations = {}
    cache_num_divisors   = {1: 1}
    candidate  = 1
    # dynamic programming execution assuming that the candidate is bounded
    # here's the trick: we don't check the number to be outputed, but its near-squarer-root.
    while candidate < 1e+6:
        candidate += 1
        factors = factorize(candidate, cache_factorizations)
        cache_num_divisors[candidate] = get_num_divisors(factors)
        # calculate the number of divisors of (candidate-1) * candidate / 2
        if candidate % 2 == 0:
            prod_num_divisors = cache_num_divisors[candidate // 2] * cache_num_divisors[candidate - 1]
        else:
            prod_num_divisors = cache_num_divisors[candidate] * cache_num_divisors[(candidate - 1) // 2]
        if prod_num_divisors > n:
            return prod_num_divisors, int((candidate - 1) * candidate / 2)
    return -1        

def Euler_Problem_13():
    '''
    Work out the first ten digits of the sum of the following one-hundred 50-digit numbers.
    '''
    raw_list = '''
    37107287533902102798797998220837590246510135740250
    46376937677490009712648124896970078050417018260538
    74324986199524741059474233309513058123726617309629
    91942213363574161572522430563301811072406154908250
    23067588207539346171171980310421047513778063246676
    89261670696623633820136378418383684178734361726757
    28112879812849979408065481931592621691275889832738
    44274228917432520321923589422876796487670272189318
    47451445736001306439091167216856844588711603153276
    70386486105843025439939619828917593665686757934951
    62176457141856560629502157223196586755079324193331
    64906352462741904929101432445813822663347944758178
    92575867718337217661963751590579239728245598838407
    58203565325359399008402633568948830189458628227828
    80181199384826282014278194139940567587151170094390
    35398664372827112653829987240784473053190104293586
    86515506006295864861532075273371959191420517255829
    71693888707715466499115593487603532921714970056938
    54370070576826684624621495650076471787294438377604
    53282654108756828443191190634694037855217779295145
    36123272525000296071075082563815656710885258350721
    45876576172410976447339110607218265236877223636045
    17423706905851860660448207621209813287860733969412
    81142660418086830619328460811191061556940512689692
    51934325451728388641918047049293215058642563049483
    62467221648435076201727918039944693004732956340691
    15732444386908125794514089057706229429197107928209
    55037687525678773091862540744969844508330393682126
    18336384825330154686196124348767681297534375946515
    80386287592878490201521685554828717201219257766954
    78182833757993103614740356856449095527097864797581
    16726320100436897842553539920931837441497806860984
    48403098129077791799088218795327364475675590848030
    87086987551392711854517078544161852424320693150332
    59959406895756536782107074926966537676326235447210
    69793950679652694742597709739166693763042633987085
    41052684708299085211399427365734116182760315001271
    65378607361501080857009149939512557028198746004375
    35829035317434717326932123578154982629742552737307
    94953759765105305946966067683156574377167401875275
    88902802571733229619176668713819931811048770190271
    25267680276078003013678680992525463401061632866526
    36270218540497705585629946580636237993140746255962
    24074486908231174977792365466257246923322810917141
    91430288197103288597806669760892938638285025333403
    34413065578016127815921815005561868836468420090470
    23053081172816430487623791969842487255036638784583
    11487696932154902810424020138335124462181441773470
    63783299490636259666498587618221225225512486764533
    67720186971698544312419572409913959008952310058822
    95548255300263520781532296796249481641953868218774
    76085327132285723110424803456124867697064507995236
    37774242535411291684276865538926205024910326572967
    23701913275725675285653248258265463092207058596522
    29798860272258331913126375147341994889534765745501
    18495701454879288984856827726077713721403798879715
    38298203783031473527721580348144513491373226651381
    34829543829199918180278916522431027392251122869539
    40957953066405232632538044100059654939159879593635
    29746152185502371307642255121183693803580388584903
    41698116222072977186158236678424689157993532961922
    62467957194401269043877107275048102390895523597457
    23189706772547915061505504953922979530901129967519
    86188088225875314529584099251203829009407770775672
    11306739708304724483816533873502340845647058077308
    82959174767140363198008187129011875491310547126581
    97623331044818386269515456334926366572897563400500
    42846280183517070527831839425882145521227251250327
    55121603546981200581762165212827652751691296897789
    32238195734329339946437501907836945765883352399886
    75506164965184775180738168837861091527357929701337
    62177842752192623401942399639168044983993173312731
    32924185707147349566916674687634660915035914677504
    99518671430235219628894890102423325116913619626622
    73267460800591547471830798392868535206946944540724
    76841822524674417161514036427982273348055556214818
    97142617910342598647204516893989422179826088076852
    87783646182799346313767754307809363333018982642090
    10848802521674670883215120185883543223812876952786
    71329612474782464538636993009049310363619763878039
    62184073572399794223406235393808339651327408011116
    66627891981488087797941876876144230030984490851411
    60661826293682836764744779239180335110989069790714
    85786944089552990653640447425576083659976645795096
    66024396409905389607120198219976047599490197230297
    64913982680032973156037120041377903785566085089252
    16730939319872750275468906903707539413042652315011
    94809377245048795150954100921645863754710598436791
    78639167021187492431995700641917969777599028300699
    15368713711936614952811305876380278410754449733078
    40789923115535562561142322423255033685442488917353
    44889911501440648020369068063960672322193204149535
    41503128880339536053299340368006977710650566631954
    81234880673210146739058568557934581403627822703280
    82616570773948327592232845941706525094512325230608
    22918802058777319719839450180888072429661980811197
    77158542502016545090413245809786882778948721859617
    72107838435069186155435662884062257473692284509516
    20849603980134001723930671666823555245252804609722
    53503534226472524250874054075591789781264330331690
    '''
    # take the first 16 digits and throw away the rest.
    arr_in_str_form = [_x for _x in raw_list.split('\n') if len(_x.replace(' ', '')) > 0]
    arr = [int(_y[:16]) for _y in arr_in_str_form]
    assert len(arr) == 100

    return str(sum(arr))[:10]

def Euler_Problem_14(n=1000000):
    '''
    The following iterative sequence is defined for the set of positive integers:
    n -> n/2 (n is even)
    n -> 3n + 1 (n is odd)
    Using the rule above and starting with 13, we generate the following sequence:
    13 -> 40 -> 20 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1
    It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.
    Which starting number, under one million, produces the longest chain?
    NOTE: Once the chain starts the terms are allowed to go above one million.
    '''
    def collatz(num):
        '''
        Subroutine to calculate the next term in the collatz sequence.
        '''
        assert num > 0 and isinstance(num, int)
        next_term = (num // 2) if (num % 2 == 0) else (3 * num + 1)
        return next_term

    def lookup_collatz_length(num, cache):
        ''' 
        Uses recursion with cache for "non-sequential dynamic programming".
        '''
        assert cache[1] == 1
        if num in cache.keys():
            return cache[num]
        else:
            length = lookup_collatz_length(collatz(num), cache) + 1
            cache[num] = length
            return length

    # initialize cache
    solution_cache = {1: 1}
    max_length = 1
    best_start = 1
    for k in range(1, n):
        _length = lookup_collatz_length(k, solution_cache)
        if _length > max_length:
            max_length = _length
            best_start = k
    # reconstuct the longest collatz sequence        
    best_chain = []
    value = best_start
    while value > 1:
        best_chain.append(value)
        value = collatz(value)
    best_chain.append(1)
    return best_start, max_length, best_chain

def Euler_Problem_15(n=20):
    '''
    Starting in the top left corner of a 2 * 2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.
    How many such routes are there through a 20 * 20 grid?
    '''
    # this is classic "2n choose n".
    num_routes = 1.0
    # trick to compute (2n)! / n! with less risk of numeric overflow
    for k in range(1, n+1):
        num_routes *= (n+k) / k
    return round(num_routes)

def Euler_Problem_16(n=1000):
    '''
    2 ^ 15 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
    What is the sum of the digits of the number 2 ^ 1000?
    '''
    # 2 ^ 1000 is between 10^250 and 10^333, which is too large for typical data types.
    # However, one can easily store by digit.
    def multiply_by_2(orderDict):
        '''
        Subroutine to multiply a number by 2.
        orderDict -- defaultdict of magnitude -> value mapping.
        Eg. {0: 1, 1: 3, 2: 6} stands for 1*10^1 + 3*10^1 + 6*10^2 = 631.
        '''
        from collections import defaultdict
        retDict = defaultdict(int)
        for _key, _value in orderDict.items():
            doubled = _value * 2
            current = doubled % 10
            incremt = doubled // 10
            retDict[_key] += current
            if incremt > 0:
                retDict[_key+1] += incremt
        return retDict
    
    # run subroutine n times
    num = {0: 1}
    for k in range(0, n):
        num = multiply_by_2(num)
    return sum(num.values())  

def Euler_Problem_17(n=1000):
    '''
    If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.
    If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?
    NOTE: Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. The use of "and" when writing out numbers is in compliance with British usage.
    '''
    hard_coded_mapping = {
            0: 0, # "zero" in this case is really "nothing"
            1: 3, # "one"
            2: 3, # "two"
            3: 5, # etc.
            4: 4,
            5: 4,
            6: 3,
            7: 5,
            8: 5,
            9: 4,
            10: 3,
            11: 6,
            12: 6,
            13: 8,
            14: 8,
            15: 7,
            16: 7,
            17: 9,
            18: 8,
            19: 8,
            20: 6,
            30: 6,
            40: 5,
            50: 5,
            60: 5,
            70: 7,
            80: 6,
            90: 6
            }
    def subroutine_small(num):
        '''
        Assumes that the number is below a hundred.
        '''
        assert isinstance(num, int) and (num >= 0) and (num < 100), num
        if num in hard_coded_mapping.keys():
            return hard_coded_mapping[num]
        else:
            floor_to_ten = (num // 10) * 10
            modulos_ten  = num % 10
            return hard_coded_mapping[floor_to_ten] + hard_coded_mapping[modulos_ten]

    def count_digits(num):
        '''
        Assumes that the number is below a million.
        '''
        assert isinstance(num, int) and (num > 0), num
        num_thousands = num // 1000
        num_hundreds  = (num % 1000) // 100
        num_small     = num % 100

        part_below_hundred = subroutine_small(num_small)
        digits = part_below_hundred
        if num_thousands > 0:
            digits += count_digits(num_thousands) + 8 # 8 from "thousand"
        if num_hundreds > 0:
            digits += count_digits(num_hundreds) + 7 # 7 from "hundred"
        if (num_thousands > 0 or num_hundreds > 0) and part_below_hundred > 0:
            digits += 3 # 3 from "and"
        return digits
    
    def test():
        from random import randint
        for k in range(0, 10):
            trial = randint(1, 10000)
            print(trial, count_digits(trial))
        for k in range(1, 20):
            trial = k * 50
            print(trial, count_digits(trial))

    total_digits = 0
    for k in range(1, n+1):
        val = count_digits(k)
        #print(k, val)
        total_digits += val
    return total_digits

def Euler_Problem_18(row_idx=-1):
    '''
    By starting at the top of the triangle below and moving to adjacent numbers on the row below, the maximum total from top to bottom is 23.
    
       3
      7 4
     2 4 6
    8 5 9 3
    That is, 3 + 7 + 4 + 9 = 23.
    Find the maximum total from top to bottom of the triangle below:
                  75
                 95 64
                17 47 82
               18 35 87 10
              20 04 82 47 65
             19 01 23 75 03 34
            88 02 77 73 07 63 67
           99 65 04 28 06 16 70 92
          41 41 26 56 83 40 80 70 33
         41 48 72 33 47 32 37 16 94 29
        53 71 44 65 25 43 91 52 97 51 14
       70 11 33 28 77 73 17 78 39 68 17 57
      91 71 52 38 17 14 91 43 58 50 27 29 48
     63 66 04 68 89 53 67 30 73 16 69 87 40 31
    04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
    NOTE: As there are only 16384 routes, it is possible to solve this problem by trying every route. However, Problem 67, is the same challenge with a triangle containing one-hundred rows; it cannot be solved by brute force, and requires a clever method! ;o)
    '''
    inp_arr = '''
    75
    95 64
    17 47 82
    18 35 87 10
    20 04 82 47 65
    19 01 23 75 03 34
    88 02 77 73 07 63 67
    99 65 04 28 06 16 70 92
    41 41 26 56 83 40 80 70 33
    41 48 72 33 47 32 37 16 94 29
    53 71 44 65 25 43 91 52 97 51 14
    70 11 33 28 77 73 17 78 39 68 17 57
    91 71 52 38 17 14 91 43 58 50 27 29 48
    63 66 04 68 89 53 67 30 73 16 69 87 40 31
    04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
    '''
    arr = [[int(_z) for _z in _y.split(' ') if len(_z) > 0] for _y in inp_arr.split('\n')]
    arr = [_l for _l in arr if len(_l) > 0]

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

def Euler_Problem_19(n=2000):
    '''
    You are given the following information, but you may prefer to do some research for yourself.
    1 Jan 1900 was a Monday.
    Thirty days has September,
    April, June and November.
    All the rest have thirty-one,
    Saving February alone,
    Which has twenty-eight, rain or shine.
    And on leap years, twenty-nine.
    A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.
    How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?
    '''
    month_to_days_common = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    month_to_days_leap = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    def count_Sunday_1sts(year, first_day):
        '''
        Subroutine to count the number of 1st Sundays in a year.
        '''
        # set up calculation
        month_to_days = dict(month_to_days_common)
        if year % 4 == 0:
            if (year % 100 != 0) or (year % 400 == 0):
                month_to_days = dict(month_to_days_leap)
        val = first_day

        # loop over months
        count = 0
        months = []
        for _month in range(1, 13):
            if val % 7 == 0:
                count += 1
                months.append((year, _month))
            val += month_to_days[_month]
        return count, val % 7, months[:]

    # Jan 1 1900 was a Monday
    first_day = 1
    total_count = 0
    match_months = []
    for _year in range(1900, n+1):
        count, first_day, months = count_Sunday_1sts(_year, first_day)
        total_count += count
        match_months += months
    # the problem asks for Jan 1, 1901 to Dec 31, 2000, so we exclude 1900
    return total_count - count_Sunday_1sts(1900, 1)[0]

def Euler_Problem_20(n=100):
    '''
    n! means n * (n - 1) * ... * 3 * 2 * 1

    For example, 10! = 10 * 9 * ... * 3 * 2 * 1 = 3628800,
    and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.
    
    Find the sum of the digits in the number 100!
    '''
    # 100! is between 10^100 and 10^200, which is too large for typical data types.
    # However, one can easily store by digit.
    from collections import defaultdict
    def multiply_by_constant(orderDict, c):
        '''
        Subroutine to multiply a number by c.
        orderDict -- defaultdict of magnitude -> value mapping.
        Eg. {0: 1, 1: 3, 2: 6} stands for 1*10^1 + 3*10^1 + 6*10^2 = 631.
        '''
        retDict = defaultdict(int)
        for _key, _value in orderDict.items():
            multiplied = _value * c
            shift = 0
            while multiplied > 0 or shift == 0:
                retDict[_key+shift] += (multiplied % 10)
                multiplied = multiplied // 10
                shift += 1
        return retDict

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

    # run subroutine n times
    num = {0: 1}
    for k in range(1, n+1):
        num = multiply_by_constant(num, k)
        num = correct_digits(num)
    return sum(num.values())  






