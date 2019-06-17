from lightning._build_utils import maybe_cythonize_extensions
import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('linear_model', parent_package, top_path)

    config.add_extension('loss_fast',
                         sources=['loss_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])
    config.add_extension('cd_primal_sparse_mb',
                         sources=['cd_primal_sparse_mb.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    maybe_cythonize_extensions(top_path, config)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())