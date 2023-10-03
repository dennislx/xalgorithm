__all__ = ['PyVersion', 'csv', 'time']

from IPython import get_ipython 
from IPython.core.magic import ( Magics, magics_class, line_magic, cell_magic, line_cell_magic)
from IPython.core.magic_arguments import (magic_arguments, argument, parse_argstring)
from types import ModuleType
from sys import version_info as V
from io import StringIO

from xalgorithm.out.pd_rich import print_df, pd
from xalgorithm.utils import xprint
import time as T
import argparse

PYTHON_VER = '.'.join(map(str, [V.major, V.minor, V.micro]))

def print_versions(symbol_table=locals()):
    for val in symbol_table.values():
        if isinstance(val, ModuleType):
            try: print('{:>10}  {}'.format(val.__name__, val.__version__))
            except AttributeError: continue
    xprint(f'[b]Python[green] {PYTHON_VER}[/green][/b]')

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
@argument('title', nargs='*', default=["time"], help=("the title of this cell execution"), action=ParseString)
@cell_magic
def time(line, cell):
    args = parse_argstring(time, line)
    start = T.time()
    get_ipython().run_cell(cell)
    xprint("[b][green]%s[/green]: [cyan]%s[/cyan] seconds[/b]" % (args.title, T.time() - start))


@magic_arguments()
@argument('-f', '--format', default='rich', choices=['rich', 'markdown', 'plain'])
@argument('-s', '--sep', default=',', type=str, help=("the delimiter that separates the values, sep is set to comma by default"))
@cell_magic
def csv(line, cell):
    r"""Parses the cell into a DataFrame and then prints the DataFrame to the console in the specified format.
    
    warning: please remember to put deinlimiter in double quote string
    """
    args = parse_argstring(csv, line)
    sio  = StringIO(cell)
    df   = pd.read_csv(sio, sep=',', skipinitialspace=True)
    if args.format == 'plain':
        return df
    elif args.format == 'markdown':
        headers = [x + ' &nbsp;' for x in df.columns]
        kwargs = dict(index=False, numalign="left", headers=headers)
        return print(df.to_markdown(**kwargs))
    elif args.format == 'rich':
        return print_df(df)
    else:
        raise NotImplementedError("I haven't implemented handler to deal with this format yet")
    

if __name__ == '__main__':
    str = """
    tag, input, effects
    b, [b]test[/b], boldface text
    i, [i]test[/i], display the content in italic style
    u, [u]test[/u], underline the text
    s, [s]test[/s], draws a horizontal line over the text
    dim, [dim]test[/dim], decrease contrast of text
    fg, [fg red]test[/fg], color text in red
    bg, "[bg 255,0,0]test[/bg]", set background color of text to red
    """
    sio  = StringIO(str)
    df   = pd.read_csv(sio, sep=',', skipinitialspace=True)
    print_df(df)
