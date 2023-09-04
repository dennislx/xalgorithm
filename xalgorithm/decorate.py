__all__ = ["print_me", "record_it", "ctrl_c", "tag_me"]

from functools import wraps, partial, lru_cache
from logging import getLogger, Logger
from typing import Callable, Optional
from time import perf_counter
from signal import signal, SIGINT, SIGHUP, SIGTERM

default_logger = getLogger(__name__)

@lru_cache(maxsize=None)  
def tag_me(tagName: str):
    """
    The function `get_func_tag` returns a decorator that can be used to tag other functions with a specified tag name.

    ```
    dp = tag_me('dynamic programming')
    @dp
    def fA(*args, **kwargs):
        print("A", args, kwargs)
    ```
    >>> dp.invoke('fA', 'hello', next_word = 'world')
    >>> A ('hello',) {'next_word': 'world'}
    """
    return Tag(tagName)

class Tag(object):
    def __init__(self, tagName):
        self.functions = {}
        self.tagName = tagName
    def __str__(self):
        return f"<Tag {self.tagName}>"
    def __call__(self, f):
        self.functions[f.__name__] = f
        return f
    def invoke(self, func_name, *args, **kwargs):
        if func_name in self.functions:
            return self.functions[func_name](*args, **kwargs)
        else:
            raise ValueError(f"Function '{func_name}' not found in {self.tagName}")

def ctrl_c(func):
    """
    The function prompts the user to enter "yes" or "y" to quit the program, and "no" or "n" to continue when you hit CTRL + C in the console
    """
    signal(SIGINT, _ask_or_quit)
    signal(SIGHUP, _ask_or_quit)
    signal(SIGTERM, _ask_or_quit)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def _ask_or_quit(_signal, _frame):
    while True:
        exit_flag = input("Type yes or y to quit the program? (yes|y) / (no|n)\n>> ")
        if exit_flag in {"yes", "y"}:
            default_logger.warning(">> exit...")
            exit()
        elif exit_flag in {"no", "n"}:
            break


def record_it(func: Optional[Callable] = None, *, logger: Logger = default_logger, name: Optional[str] = None, stat: str = 'time') -> Callable:
    """
    The `record_it` function is a decorator that measures either the execution time or the execution count of a function and print it in the logger

    ```
    @record_it(type='time', name='test_time')
    def heavy(num): 
        x = 0
        for _ in range(num): x += _
    ```
    >>> heavy(1000000)
    """
    if stat not in ('time', 'count'):
        raise ValueError("stat (statistics) should be either time or count")
    if func is None: return partial(record_it, logger=logger, name=name, stat=stat)
    if name is None: name = func.__name__
    if stat == 'time':
        return _run_time(func, logger, name)
    else:
        return _run_count(func, logger, name)
    
def _run_count(func, logger, name):
    count = 0   # global variable
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal count
        count += 1
        logger.warning(f"[{name}]: called {count} times")
        return func(*args, **kwargs)
    return wrapper

def _run_time(func, logger, name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        _return = func(*args, **kwargs)
        cost_time = round(perf_counter() - start_time, 5)
        logger.warning(f"[{name}]: {cost_time} seconds")
        return _return
    return wrapper

def print_me(func:Callable) -> Callable:
    """
    Prints the function, it's arguments and result. Really helpful for debugging.

    ```
    @print_me
    def add(x, y): return x+y
    ```
    >>> add(2, 3)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if len(kwargs) > 0:
            print("\n{}{}{} = {}".format(func.__name__, args, kwargs, result))
        else:
            print("\n{}{} = {}".format(func.__name__, args, result))
        return result
    return wrapper