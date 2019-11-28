from sklearn._build_utils import maybe_cythonize_extensions
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('pyrfm', parent_package, top_path)
    config.add_extension('dataset_fast',
                         sources=['dataset_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    config.add_extension('kernels_fast',
                         sources=['kernels_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])
    config.add_subpackage('sfmt')
    config.add_subpackage('ffht')
    config.add_subpackage('random_feature')
    config.add_subpackage('linear_model')

    maybe_cythonize_extensions(top_path, config)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
