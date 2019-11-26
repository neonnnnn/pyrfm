import sys
import numpy
from os.path import join


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('sfmt', parent_package, top_path)
    
    config.add_extension('sfmt',
                         sources=['sfmt.pyx',
                                  join('.', 'SFMT-src-1.5.1', 'SFMT.c')],
                         include_dirs=[join('.', 'SFMT-src-1.5.1'),
                                       numpy.get_include()],
                         language='c++')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())