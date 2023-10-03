import os
import re
from typing import List, Type, TypeVar, Iterable, Any, Generic
import random
import bbcode
import numpy as np
from colorama import Fore, Style
from sty import fg, bg, ef, rs

__all__ = [
    'opath',
    'ojoin',
    'ofind',
    'oexists',
    'osplit',
    'osimplify',
    'ocode',
    'seed_all',
    'xcolor',
    'xprint'
]


class Xcolor:
    """Parse bbcode formatted string 
    ### supported formats

    | tag &nbsp;   | input &nbsp;          | effects &nbsp;                        |
    |:-------------|:----------------------|:--------------------------------------|
    | b            | [b]test[/b]           | boldface text                         |
    | i            | [i]test[/i]           | display the content in italic style   |
    | u            | [u]test[/u]           | underline the text                    |
    | s            | [s]test[/s]           | draws a horizontal line over the text |
    | dim          | [dim]test[/dim]       | decrease contrast of text             |
    | fg           | [fg red]test[/fg]     | color text in red                     |
    | bg           | [bg 255,0,0]test[/bg] | set background color of text to red   |`
    """
    def __init__(self):
        self.parser = bbcode.Parser(install_defaults=False, escape_html=False, replace_cosmetic=False, replace_links=False, newline='\n')
        self.install_default_formatters()

    def install_default_formatters(self):
        effects = dict(
            b=(ef.b, rs.bold_dim), i=(ef.i, rs.i), u=(ef.u, rs.u),
            s=(ef.strike, rs.strike), dim=(ef.dim, rs.dim_bold)
        )
        for name, (bf,af) in effects.items():
            self.parser.add_simple_formatter(name, "{}%(value)s{}".format(bf, af))
        colors = {'red', 'green', 'blue', 'cyan', 'magenta', 'black', 'white', 'grey'}
        for c in colors: 
            self.parser.add_simple_formatter(c, '{}%(value)s{}'.format(fg(c), rs.fg))
        def render_color(tag_name, value, options, parent, context):
            color = ' '.join([k for k in options.keys()])
            f = eval(tag_name)
            g = lambda _: '{}{}{}'.format(_, value, rs(tag_name))
            return g(f(*eval(color))) if ',' in color else g(f(color))
        self.parser.add_formatter('fg', render_color, strip=True, swallow_trailing_newline=True)
        self.parser.add_formatter('bg', render_color, strip=True, swallow_trailing_newline=True)
        

    def __call__(self, string):
        return self.parser.format(string)

STY = Xcolor()

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)

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

def ojoin(*args, create_if_not_exist=False, expand_user=False):
    """
    The `ojoin` function joins multiple path components and optionally creates the path if it does not exist.
    
    :return: the joined path.
    """
    path = os.path.join(*args)
    if create_if_not_exist: omake(path)
    if expand_user: path = os.path.expanduser(path)
    return path

def ocode(file):
    """Launch VSCode to open the file"""
    os.system(f"code {file}")

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

def xcolor(string, color="cyan", bold=False, display=False):
    r"""Returns stylized string with coloring and bolding for printing.
    >>> print(xcolor('hello world', 'green', bold=True))
    """
    colors = {'red': Fore.RED, 'green': Fore.GREEN, 'blue': Fore.BLUE, 'cyan': Fore.CYAN, 'magenta': Fore.MAGENTA, 'black': Fore.BLACK, 'white': Fore.WHITE}
    style = colors[color]
    if bold: style += Style.BRIGHT
    out = style + string + Style.RESET_ALL
    if display: print(out)
    return out

def xprint(string, display=True):
    r"""Output styled text by bridging the gap between 'sty' and 'bbcode' to avoid long-winded processes in rich.print implementation

    ### STY
    ef(effect): bold, italic, underl, inverse, dim, hidden, strike, rs
    bg(background)/fg(foreground): black, da_grey, grey, li_grey, magenta, cyan, rs, we can also do fg(10, 255, 10) to define color in rgb format

    ### Example
    ```
    >>> xprint("[b]boldface text[/b]")
    ```

    ### TODO
    The current implementation doesn't support nested BBCode within the same category. Consequently, in the example provided:
    ```
    >>> [red]red [blue]blue [green]green[/green] red[/red] 
    ```
    the last word enclosed in `[red]` tags won't be colored in red as expected.
    """
    out = STY(string)
    if display: print(out)
    else: return out

    
if __name__ == '__main__':
    xprint("[green][fg red][blue]I[/blue] really [cyan]like[/cyan][/fg] Python[/green]")
