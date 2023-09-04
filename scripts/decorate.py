import logging
from xalgorithm import record_it, ctrl_c, tag_me

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
def fA(*args, **kwargs):
    print("A", args, kwargs)

def test_tagme():
    who = tag_me('dynamic programming')
    who.invoke('fA', 'hello', next_word = 'world')


