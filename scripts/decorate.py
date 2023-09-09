import logging
from xalgorithm import record_it, ctrl_c, tag_me, print_me
from functools import lru_cache

logger = logging.getLogger(__name__)

@record_it(stat='time', name="timing function")
@record_it(stat='count', name="count function")
@ctrl_c
def calculate_million_numbers(num):
    number = 0
    for _ in range(num):
        number += 1

def test_recording():
    for index in range(5):
        calculate_million_numbers(1000000)


@tag_me('dynamic programming')
@lru_cache()
def fib(n):
    if n <= 2: return 1
    return fib(n-1) + fib(n-2)

def test_tagme():
    who = tag_me('dynamic programming')
    res = who.invoke('fib', n=5)
    print(res)

test_tagme()

@print_me
def add(x, y):
    return x + y

