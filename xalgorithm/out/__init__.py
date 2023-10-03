from sql.magic import SqlMagic
from .magic_cmd import *
from .pd_rich import *
# from .chatgpt import *

__all__ = []
__all__.extend(pd_rich.__all__)
__all__.extend(magic_cmd.__all__)


def load_ipython_extension(ipython):
    """Load the extension in IPython.
    
    NOTE: we have to somehow execute @magicclass before any @magicfunction to avoid execution error. This is to do with how it implements magics_class in core/magic.py
    """
    ipython.register_magics(SqlMagic, PyVersion)
    ipython.register_magic_function(csv, magic_kind='cell',  magic_name='csv')
    ipython.register_magic_function(time, magic_kind='cell',  magic_name='time')
    lines = ['%py_version', '%sql']
    cells = ['%%csv', '%%time', '%%sql']
    print("Line Magic: {} \nCell Magic: {}".format(lines, cells))