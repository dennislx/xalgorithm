from .pd_rich import *
from .magic_cmd import *
# from .chatgpt import *

__all__ = []
__all__.extend(pd_rich.__all__)
__all__.extend(magic_cmd.__all__)


def load_ipython_extension(ipython):
    """Load the extension in IPython."""
    ipython.register_magics(PyVersion)
    ipython.register_magic_function(csv, magic_kind='cell',  magic_name='csv')
    ipython.register_magic_function(time, magic_kind='cell',  magic_name='time')
    lines = ['%py_version']
    cells = ['%%csv', '%%time']
    print("Line Magic: {} \nCell Magic: {}".format(lines, cells))