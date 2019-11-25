import sys
import numpy
from os.path import join


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('ffht', parent_package, top_path)
    
    config.add_extension('fht_fast',
                         sources=['fht_fast.pyx',
                                  join('.', 'FFHT', 'fht_avx.c')],
                         include_dirs=[join('.', 'FFHT'),
                                       numpy.get_include()],
                         depends=[join('.', 'FFHT', 'fht.h')])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())