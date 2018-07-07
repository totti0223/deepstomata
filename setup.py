"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(

    name='deepstomata',
    version='1.0',
    description='iterative calculation of stomatal pores ',
    long_description=long_description,
    url='https://github.com/totti0223/',
    author='Yosuke Toda',
    author_email='tyosuke@aquaseerser.com',
    license='MIT',

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ],
    keywords='Facial Recognition Technology for Automated Stomatal Aperture Measurement',
    packages=['deepstomata'],
    install_requires=open('requirements.txt').read().splitlines(),
    package_data={
        "deepstomata": ["models/*", "config.ini"]
    }
)
