# pip install multipledispatch

from multipledispatch import dispatch

@dispatch(int, int, int)
def add(a, b, c): # type: ignore
    print(a + b + c)

@dispatch(int, int)
def add(a, b): # type: ignore
    print(a + b)

@dispatch(int, str)
def add(a, b):
    print(a + b)

def fn_dispatch():
    add(1, 2)
    add(1, 2, 3) # type: ignore
    add("hello", " world")
    add("hello", " world", " !") # type: ignore

def fn_cython():
    import pyximport
    pyximport.install(reload_support=True, language_level=3)
    from xalgorithm.hpc import do_Cprime, do_prime
    import time
    def run(name, fun): 
        start = time.time()
        print(fun(500)[-5:])
        print(f"{name} time: {time.time() - start} sec\n")
    run("Cython", do_Cprime.primes)
    run("Python", do_prime.primes)

fn_cython()