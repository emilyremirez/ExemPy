from setuptools import setup, find_packages

setup(
    name = 'ExemPy',
    version = '0.1',
    description = 'Routines for simulating the Generalized Context Model of speech perception',
    author = 'Emily Remirez',
    packages = ['ExemPy','ExemPy.GCM','ExemPy.utils','ExemPy.viz'],
    install_requires=[
        'importlib_resources; python_version <"3.9"',
        'math',
        'random',
        'numpy',
        'scipy',
        'matplotlib',
        'seaborn'
    ]
)