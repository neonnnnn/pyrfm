from __future__ import print_function
import os.path
import sys
import setuptools
from numpy.distutils.core import setup

DISTNAME = 'randfeature'
DESCRIPTION = 'A python library for random feature maps.'
VERSION = '0.0.dev0'

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('randfeature')

    return config

if __name__ == '__main__':
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer='neonnnnn',
          include_package_data=True,
          version=VERSION,
          zip_safe=False,
          )
