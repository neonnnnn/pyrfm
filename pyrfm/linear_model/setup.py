from lightning._build_utils import maybe_cythonize_extensions
import numpy
from os.path import join


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('linear_model', parent_package, top_path)

    sfmtdir = join(top_path, 'pyrfm', 'sfmt', 'SFMT-src-1.5.1')
    config.add_extension('loss_fast',
                         sources=['loss_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    config.add_extension('utils', sources=['utils.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])

    config.add_extension('cd_primal_sparse_mb',
                         sources=['cd_primal_sparse_mb.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include()])

    config.add_extension('adagrad_fast',
                         sources=['adagrad_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])

    config.add_extension('adam_fast',
                         sources=['adam_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])

    config.add_extension('sdca_fast', sources=['sdca_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])

    config.add_extension('sgd_fast', sources=['sgd_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])

    config.add_extension('saga_fast', sources=['saga_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])
    
    config.add_extension('doubly_sgd_fast', sources=['doubly_sgd_fast.pyx'],
                         language='c++',
                         include_dirs=[numpy.get_include(), sfmtdir])

    maybe_cythonize_extensions(top_path, config)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())