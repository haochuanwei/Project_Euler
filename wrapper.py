import time
def timeit(method):
    def timed(*args, **kw):
        tic = time.time()
        result = method(*args, **kw)
        toc = time.time()
        print('Timeit: {0} took {1} ms.'.format(method.__name__, (toc - tic) * 1000))
        return result
    return timed
