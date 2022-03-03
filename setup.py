from setuptools import setup

setup(
    name="jijbench",
    python_requires=">=3.8, <3.11",
    packages=["jijbench", "jijbench.problems"],
    version_config={
        "template": "{tag}",
        "dirty_template": "{tag}",
    },
    setup_requires=[
        "setuptools-git-versioning",
    ],
    install_requires=[
        "openjij",
        "numpy",
        "pandas",
        "matplotlib",
    ],
)
