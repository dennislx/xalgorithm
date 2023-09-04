from functools import lru_cache
from collections import defaultdict, deque
from typing import List, Iterable, TypeVar, Union, Generator
from xalgorithm.utils import isIterable
from xalgorithm.decorate import tag_me

__all__ = [
    'freq_lt_ntimes',
    'flatten_nested_array'
]

T = TypeVar('T')

def freq_lt_ntimes(array: List[int], n: int, naive=False) -> List[int]:
    """
    Given a list lst and a number N, create a new list that contains each number of the list at most N times without reordering.

    >>> array = [1,2,3,1,2,1,2,3]; n = 2
    >>> freq_lt_ntimes(array, n) == [1,2,3,1,2]
    """
    if naive: return _freq_lt_ntimes(array, n)
    ans = []
    counts = defaultdict(int)  # keep track of occurrences
    for i in array:
        if counts[i] < n:
            ans.append(i)
            counts[i] += 1
    return ans


def _freq_lt_ntimes(array, n):
    """
    The `_freq_lt_ntimes` function returns a list of numbers from the input array that occur less than `n` times.
    """
    ans = []
    for num in array:
        if ans.count(num) < n:
            ans.append(num)
    return ans

def flatten_nested_array(array: List[T], return_iterator = False) -> Union[List[T], Generator]:
    """
    Implement Flatten Arrays.  Given an array that may contain nested arrays, produce a single resultant array. If `return_iterator` is set to `True`, the function will return a generator.
    
    >>> array = [[], [8]]
    >>> flatten_nested_array(array) == [8]
    """
    if return_iterator: return _flatten_nested_array(array)
    ans = []
    for sub in array:
        if isIterable(sub):
            ans.extend(flatten_nested_array(sub)) # type: ignore
        else:
            ans.append(sub)
    return ans

def _flatten_nested_array(array):
    """
    The function `_flatten_nested_array` recursively flattens a nested array into a single-dimensional array.
    """
    for sub in array:
        if isIterable(sub):
            yield from _flatten_nested_array(sub)    
        else:
            yield sub

@tag_me("Greedy")
@tag_me("Breadth First Search")
def hua_rong_dao(initial: List[int], final: List[int], use_bfs = False) -> int:
    """
    Determine the minimum steps required to rearrange numbers from
    initial state to the final state. You may swap ONE number with 0 each step 
    (there is only one empty spot)

    >>> initial = [1, 2, 3, 0, 4]; final = [0, 3, 2, 1, 4]
    >>> hua_rong_dao(initial, final) == 4

    - The default algorithm to solve this problem is Greedy
    - Another breadth first search solution is provided in addition 
    """
    if use_bfs:
        return _bfs_hua_rong_dao(initial, final)
    else:
        return _greedy_hua_rong_dao(initial, final)

def _greedy_hua_rong_dao(initial, final):
    """
    So, the idea is that we continuously search for the one empty spot. 

    - If the ZERO is in a spot different from the final state, swap it with where it should be
    - Otherwise, we keep searching for another spot where there's a mismatch and then swap it with ZERO.
    """
    ans, n = 0, len(initial)
    def swap(i, j):
        initial[i], initial[j] = initial[j], initial[i]
    while initial != final:
        zero = initial.index(0)
        if zero != final.index(0):
            swap_with = initial.index(final[zero])
            swap(zero, swap_with)
        else:
            for i in range(n):
                if initial[i] != final[i]:
                    swap(zero, i)
                    break
        ans += 1
    return ans

def _bfs_hua_rong_dao(initial, final):
    """
    Explore all possible game states of the puzzle and find the minimum expenses
    to reach final state from initial state
    """
    Q = deque([(initial, 0)])
    def hash(lst):
        return ''.join(map(str, lst))
    def get_next(state):
        zero = state.index(0)
        for i, x in enumerate(state):
            if x != 0:
                next = state.copy()
                next[i], next[zero] = next[zero], next[i]
                yield next
    answers = {hash(initial): 0} # the cost from initial to current state
    while Q:
        current, steps = Q.popleft()
        if current == final: return steps
        for next in get_next(current):
            next_code = hash(next)
            if next_code not in answers or answers[next_code] > 1 + steps:
                answers[next_code] = 1 + steps
                Q.append((next, answers[next_code])) # type: ignore
    return -1



            