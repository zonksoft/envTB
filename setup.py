import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="envtb",
    version="0.1.0",
    author="Rafael Reiter",
    author_email="mail@zonk.at",
    description="Environmental-dependent Tight-Binding package",
    license="BSD",
    keywords="solid state physics tight binding",
    url="https://github.com/zonksoft/envTB",
    packages=['envtb'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)
