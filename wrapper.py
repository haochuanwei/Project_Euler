'''
Decorators to provide bind common utilities to functions.
'''

import time

def timeit(method):
    """
    Timing decorator.
    """
    def timed(*args, **kw):
        tic = time.time()
        result = method(*args, **kw)
        toc = time.time()
        print('Timeit: {0} took {1} ms.'.format(method.__name__, (toc - tic) * 1000))
        return result
    return timed
