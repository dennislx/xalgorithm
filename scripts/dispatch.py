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

add(1, 2)
add(1, 2, 3) # type: ignore
add("hello", " world")
add("hello", " world", " !") # type: ignore