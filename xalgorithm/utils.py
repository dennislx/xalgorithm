import os
import re
from typing import List, Type, TypeVar, Iterable, Any, Generic


__all__ = [
    'opath',
    'ojoin',
    'ofind',
    'oexists',
    'osplit',
    'osimplify',
]

T = TypeVar("T")

def isType(x: T, TYPE: Type[T]):
    return isinstance(x, TYPE)

def isIterable(x: Any):
    return not isType(x, str) and isType(x, Iterable)

def r(lst):
    """
    The function returns the last element of a list if the list has only one element, otherwise it returns the entire list.
    """
    return lst[-1] if len(lst) == 1 else lst

def opath(file):
    """
    The `opath` function returns the absolute path of a file, expanding any user shortcuts.
    :return: The function `opath` returns the absolute path of the input file.
    """
    return os.path.abspath(os.path.expanduser(file))

def ojoin(*args, create_if_not_exist=False):
    """
    The `ojoin` function joins multiple path components and optionally creates the path if it does not exist.
    
    :return: the joined path.
    """
    path = os.path.join(*args)
    if create_if_not_exist: omake(path)
    return path

def omake(*args) -> List[os.PathLike]:
    """
    The `omake` function creates directories for the given file paths if they don't already exist.
    :return: The `omake` function returns a list of the directories that were created for the given file paths.
    """
    paths = []
    for path in map(os.path.dirname, args):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        paths.append(path)
    return r(paths)

def ofind(path, pattern):
    """
    The `ofind` function searches for files in a given directory that match a specified pattern and returns their paths.
    """
    pattern = re.compile(pattern)
    for _,_,file in os.walk(path):
        for each in file:
            if pattern.search(each):
                yield ojoin(path, each)

def oexists(*path) -> bool:
    """
    The function `oexists` checks if all the given paths exist.
    """
    if not path: return False
    def check_exist(path):
        return os.path.exists(path)
    return all(check_exist(p) for p in path)

def osplit(path, sep='/') -> List[str]:
    """
    The `osplit` function splits a given path into two parts based on a specified separator.
    """
    split_part = path.rpartition(sep)
    return split_part[:1] + split_part[2:]

def osimplify(path) -> os.PathLike:
    """
    The `osimplify` function takes a file path as input and returns a simplified version of the path by removing unnecessary ".." and "." tokens.

    >>> path = "/home/", => "/home"
    >>> path = "/a/./b/../../c/", => "/c"
    """
    stack, tokens = [], path.split("/")
    for token in tokens:
        if token == ".." and stack:
            stack.pop()
        elif token != ".." and token != "." and token:
            stack.append(token)
    return "/" + "/".join(stack) # type: ignore