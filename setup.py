from setuptools import setup
from setuptools import find_packages

REQUIRED_PACKAGES = ["matplotlib", "numpy", "pandas >= 0.23.0", "pymc3"]

setup(
    name="pmprophet",
    version="0.2.1",
    description="Pymc3-based universal time series prediction and decomposition library inspired by Facebook Prophet",
    url="https://github.com/luke14free/pm-prophet",
    author="Luca Giacomel",
    author_email="luca.giacomel@gmail.com",
    license="Apache License 2.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
)
