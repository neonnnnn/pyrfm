from lightning._build_utils import maybe_cythonize_extensions
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('random_feature', parent_package, top_path)

    config.add_extension('unarize',
                         sources=['unarize.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    config.add_extension('sparse_rademacher',
                         sources=['sparse_rademacher.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    config.add_extension("random_features_fast",
                         sources=['random_features_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    maybe_cythonize_extensions(top_path, config)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())