from setuptools import setup
import os


def get_readme():
    """returns the contents of the README file"""
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="pingupredictor",
    version="1.0.0",
    description="an example ML model for predicting penguin species",
    long_description=get_readme(),
    author="Kristian Rother",
    author_email="kristian.rother@posteo.de",
    packages=["..."],
    url="https://github.com/krother/ml_testing",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.11"
        "Programming Language :: Python :: 3.12"
    ],
)
