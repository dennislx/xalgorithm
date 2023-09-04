import sys, re, os, json
from setuptools import find_packages, setup

cwd = os.path.dirname(os.path.abspath(__file__))
with open(f'{cwd}/package.json', 'r') as f:
    _info = json.load(f)

version = sys.argv[1]
del sys.argv[1] # this is crucial in proceeding program run
if not re.match(r'\d\.\d\.\d', version):
    raise RuntimeError("must specify a version higher than {}".format(_info['version']))
_info['version'] = version

with open(f'{cwd}/package.json', 'w') as f:
    json.dump(_info, f)

def long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

setup(
    name=_info['name'],
    version=version,
    description='My Data Structures and Algorithms Implemented in Python',
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/dennislx/xalgorithm',
    author='Xiang Liu',
    author_email="dennisl@udel.edu",
    license='MIT',
    packages=find_packages(exclude=('tests', 'tests.*')),
    zip_safe=False,
    platforms = ["Linux"]
)