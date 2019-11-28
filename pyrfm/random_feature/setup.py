from sklearn._build_utils import maybe_cythonize_extensions
import numpy
from os.path import join
import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('random_feature', parent_package, top_path)
    sfmtdir = join(top_path, 'pyrfm', 'sfmt', 'SFMT-src-1.5.1')
    ffhtdir = join(top_path, 'pyrfm', 'ffht', 'FFHT')
    
    config.add_extension('unarize',
                         sources=['unarize.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    config.add_extension('utils_random_fast',
                         sources=['utils_random_fast.pyx'],
                         language='c++',
                         include_dirs=[sfmtdir,
                                       numpy.get_include()])

    config.add_extension("random_features_fast",
                         sources=['random_features_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(),
                                       ffhtdir])

    config.add_extension("random_features_doubly",
                         sources=['random_features_doubly.pyx'],
                         language='c++',
                         include_dirs=[sfmtdir,
                                       numpy.get_include()])

    config.add_extension("learning_kernel_with_random_feature_fast",
                         sources=["learning_kernel_with_random_feature_fast.pyx"],
                         language="c++",
                         include_dirs=[numpy.get_include()])

    config.add_subpackage("tests")

    maybe_cythonize_extensions(top_path, config)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
