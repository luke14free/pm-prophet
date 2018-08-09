from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'matplotlib',
    'numpy',
    'pandas >= 0.23.0',
    'pymc3',
]

setup(name='pmprophet',
      version='0.1',
      description='Simplified version of the Facebook Prophet model re-implemented in PyMC3 ',
      url='https://github.com/luke14free/pm-prophet',
      author='Luca Giacomel',
      author_email='',
      license='',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages())
