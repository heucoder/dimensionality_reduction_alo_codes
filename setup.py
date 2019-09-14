#coding:utf-8

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'numpy',
    'scikit-learn',
    'matplotlib',
    'tensorflow'
]

setup(
    name='feature_extract',

    version = '0.9.0',

    description = 'a set of feature extract methods',

    author = 'heucoder',

    author_email = '812860165@qq.com',

    url = 'https://github.com/heucoder/dimensionality_reduction_alo_codes/tree/reconstruct',

    install_requires = REQUIRED_PACKAGES,

    python_requires = '>=3.5',

    packages = find_packages(exclude = [])
    
)

