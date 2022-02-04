from gettext import install
from setuptools import setup

setup(
    name="jijbench",
    version="0.0.1",
    python_requires=">=3.9, <3.11",
    packages=["jijbench", "jijbench.problems"],
    install_requires=[
        "jijzept",
        "jijmodeling",
        "openjij," "numpy",
        "pandas",
        "matplotlib",
    ],
)
