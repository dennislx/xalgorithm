from typing import List
from .interface import BaseList, new_array

__all__ = ['ArrayStack', 'ControlList', 'ArrayQueue', 'ArrayDeque']

# A class that wraps a python list in the ods List interface
class ControlList(BaseList):
    def __init__(self, iterable=[]):
        self.a: List = []
        self.add_all(iterable)
    def get(self, i): return self.a[i]
    def set(self, i, x):
        y = self.a[i]; self.a[i] = x
    def add(self, i, x): self.a.insert(i, x)
    def remove(self, i): self.a.pop(i)
    def size(self): return len(self.a)

# The ArrayStack class is a subclass of the BaseList class.
class ArrayStack(BaseList):
    """Initializes an object with an array of capacity 1 and actual size 0, and adds all elements from an iterable to the object.

    #### Parameter
    @iterable It should be an iterable object, such as a list, tuple, or set.

    #### The stack elements are always stored in
        a[0], a[1], ..., a[n-1], ... a[capacity-1]
    """
    def __init__(self, iterable=[]):
        self.a = new_array(1) # capacity = 1
        self.n = 0            # actual size
        self.add_all(iterable)
    def resize(self):
        """ The `resize` function doubles the size of the array `a` and copies the elements from the original array to the new array.
        """
        b = new_array(max(1, 2*self.n))
        b[0: self.n] = self.a[0: self.n]
        self.a = b
    def get(self, i):
        """ The function `get` returns the value at index `i` in the list `a` after performing an index check.
        """
        self.check(i)
        return self.a[i]
    def set(self, i, x):
        """ The function sets the value of an element at a given index in a list.
        """
        self.check(i)
        self.a[i] = x
    def remove(self, i): 
        """ The `remove` function removes an element at index `i` from a list, updates the list size, and resizes the list if necessary.
        """
        self.check(i)
        to_pop_value = self.a[i]
        self.a[i:self.n-1] = self.a[i+1:self.n]
        self.n -= 1
        if self.capacity >= 3*self.n: 
            self.resize()
        return to_pop_value
    def add(self, i, x):
        self.check(i, ub=False)
        if self.capacity == self.n:
            self.resize()
        self.a[i+1: self.n+1] = self.a[i: self.n]
        self.a[i] = x
        self.n += 1

class ArrayQueue(BaseList):
    """a subclass of `BaseList` that represents a queue implemented using an array

    #### Parameter
    @iterable   It should be an iterable object, such as a list, tuple, or set.
    @index j    Keep track of the next element to remove
    @integer n  count the number of elements in the queue

    #### The queue elements are always stored in
        a[j % n], a[j+1 % n], ..., a[j+n-1 % n]

    #### Method
    >>> to add x, place it in a[j+n] and n++
    >>> to remove, j++ and n--
    """
    def __init__(self, iterable=[]):
        self.a = new_array(1)
        self.j, self.n = 0, 0
        self.add_all(iterable)
    def resize(self):
        """ Resizes the internal array to accommodate more elements, doubling its size if necessary, and copying existing elements.
        """
        b, capacity = new_array(max(1, 2*self.n)), self.capacity
        for k in range(self.n):
            b[k] = self.a[(self.j+k) % capacity]
        self.a, self.j = b, 0
    def add(self, x):
        """Adds an element x to the end of the queue, resizing the array if needed to accommodate the new element
        """
        if self.n + 1 > self.capacity: 
            self.resize()
        self.a[(self.j+self.n) % self.capacity] = x
        self.n += 1
    def check(self):
        if self.n == 0: raise IndexError()
    def remove(self):
        """Removes and returns the element at the front of the queue while adjusting the queue pointers and potentially resizing the array if it becomes too large.

        Gist: shrink the size and move pointer to the next of popped value
        """
        self.check()
        to_pop_value = self.a[self.j]
        self.j = (self.j + 1) % self.capacity
        self.n -= 1
        if self.capacity >= 3*self.n: self.resize()
        return to_pop_value

class ArrayDeque(BaseList):
    """The combination of ArrayStack and ArrayQueue
    """
    def __init__(self, iterable=[]):
        self.a = new_array(1)
        self.j, self.n = 0, 0
        self.add_all(iterable)
    def get_index(self, i):
        return (i+self.j) % self.capacity 
    def get(self, i):
        self.check(i)
        return self.a[ self.get_index(i) ]
    def set(self, i, x):
        self.check(i)
        self.a[ self.get_index(i) ] = x
    def add(self, i, x): 
        self.check(i, ub=False)
        if self.capacity == self.n:
            self.resize()
        if 2 * i < self.n: # left shift
            self.j = self.get_index(-1)
            for k in range(i):
                self.a[ self.get_index(k) ] = self.a[ self.get_index(k+1) ]
        else: # right shift
            for k in range(self.n, i, -1):
                self.a[ self.get_index(k) ] = self.a[ self.get_index(k-1) ]
        self.a[ self.get_index(i) ] = x
        self.n += 1
    def remove(self, i): 
        """
        to_pop_value = self.a[i]
        self.a[i:self.n-1] = self.a[i+1:self.n]
        self.n -= 1
        if self.capacity >= 3*self.n: 
            self.resize()
        """
        self.check(i)
        to_pop_value = self.a[ self.get_index(i) ]
        if 2 * i < self.n: # right shift
            for k in range(i, 0, -1):
                self.a[ self.get_index(k) ] = self.a[ self.get_index(k-1) ]
            self.j = self.get_index(1)
        else:
            for k in range(i, self.n-1): 
                self.a[ self.get_index(k) ] = self.a[ self.get_index(k+1) ]
        self.n -= 1
        if self.capacity >= 3*self.n: self.resize()
        return to_pop_value
    def resize(self):
        b = new_array(max(1, 2*self.n))
        for k in range(self.n):
            b[k] = self.a[ self.get_index(k) ]
        self.a, self.j = b, 0

if __name__ == '__main__':
    from xalgorithm.ods import ArrayStack
    arr = ArrayStack()
    arr.add(2,2)