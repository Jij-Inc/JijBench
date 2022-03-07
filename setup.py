from setuptools import setup, find_packages

setup(
    name="jijbench",
    python_requires=">=3.8, <3.11",
    packages=["jijbench"] + list(map(lambda x: "jijbench." + x, find_packages("jijbench"))),
    include_package_data=True,
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
