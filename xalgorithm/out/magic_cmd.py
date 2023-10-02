__all__ = ['PyVersion', 'csv', 'time']

from IPython import get_ipython 
from IPython.core.magic import ( Magics, magics_class, line_magic, cell_magic, line_cell_magic)
from IPython.core.magic_arguments import (magic_arguments, argument, parse_argstring)
from types import ModuleType
from sys import version_info as V
from rich import print as rprint
from io import StringIO

from xalgorithm.out.pd_rich import print_df, pd
import time as T
import argparse

PYTHON_VER = '.'.join(map(str, [V.major, V.minor, V.micro]))

def print_versions(symbol_table=locals()):
    for val in symbol_table.values():
        if isinstance(val, ModuleType):
            try: print('{:>10}  {}'.format(val.__name__, val.__version__))
            except AttributeError: continue
    rprint(f'\n[bold dark_cyan]Python {PYTHON_VER}')

class BaseMagic:
    r"""
    Base class to define my magic class
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.evaluate = True
        self.code = ''
    

@magics_class
class PyVersion(Magics):
    r""" This class has been rewritten from the [iversions](https://github.com/iamaziz/iversions)
    """
    @line_magic
    def py_version(self, line):
        print_versions(self.shell.user_ns)

class ParseString(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, ' '.join(values).capitalize())


@magic_arguments()
@argument('title', nargs='+', default=["time"], help=("the title of this cell execution"), action=ParseString)
@cell_magic
def time(line, cell):
    args = parse_argstring(time, line)
    start = T.time()
    get_ipython().run_cell(cell)
    result = "%s: %s seconds.\n" % (args.title, T.time() - start)
    rprint(result)


@magic_arguments()
@argument('-f', '--format', default='rich', choices=['rich', 'markdown', 'plain'])
@argument('-s', '--sep', default=',', type=str, help=("the delimiter that separates the values, sep is set to comma by default"))
@cell_magic
def csv(line, cell):
    r"""please remember to put delimiter in double quote string"""
    args = parse_argstring(csv, line)
    sio  = StringIO(cell)
    df   = pd.read_csv(sio, sep=',', skipinitialspace=True)
    if args.format == 'plain':
        return df
    elif args.format == 'markdown':
        headers = [x + ' &nbsp;' for x in df.columns]
        kwargs = dict(index=False, numalign="left", headers=headers)
        return print(df.to_markdown(**kwargs))
    return print_df(df)

if __name__ == '__main__':
    pass