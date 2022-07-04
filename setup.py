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
        "openjij ~= 0.5.8",
        "numpy ~= 1.23.0",
        "pandas ~= 1.4.3",
        "matplotlib ~= 3.5.2",
    ],
)
