from setuptools import setup

setup(
    name='maprf',
    version='0.0.0.dev0',
    description='Python package for Bayesian inference on parametric receptive fields.',
    url='https://github.com/mackelab/maprf',
    author='Giacomo Bassetto',
    author_email='giacomo.bassetto@gmail.com',
    classifiers=[
        'Topic :: Receptive Fields :: Bayesian inference',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='bayesian inference receptive fields parameters',
    install_requires=['numpy','scipy','matplotlib','theano'],
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
