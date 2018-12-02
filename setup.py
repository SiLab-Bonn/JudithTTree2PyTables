#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np

from subprocess import check_output

cflags = check_output(['root-config', '--cflags'])[:-1]
lflags = check_output(['root-config', '--ldflags', '--glibs'])[:-1]

extensions = [
    Extension(
        'JudithTTree2pyBARPyTables.converter',
        sources=['JudithTTree2pyBARPyTables/converter.pyx'],
        language="c++",
        extra_compile_args=cflags.split(),
        extra_link_args=lflags.split())
]

version = '1.0.0'
author = 'Jens Janssen, Christian Bespin'
author_email = 'janssen@physik.uni-bonn.de, christian.bespin@uni-bonn.de'

install_requires = ['cython', 'numpy', 'tables']

setup(
    name='JudithTTree2pyBARPyTables',
    version=version,
    description='Converting Judith ROOT TTree to pyBAR PyTables',
    url='',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    keywords=['ROOT', 'Numpy', 'Converter', 'PyTables', 'HDF5'],
    platforms='any'
)
