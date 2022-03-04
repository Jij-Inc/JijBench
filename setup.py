from setuptools import setup, find_packages
print(find_packages("jijbench"))
setup(
    name="jijbench",
    python_requires=">=3.8, <3.11",
    packages=find_packages("jijbench"),
    package_dir={"": "jijbench"},
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
