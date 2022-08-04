import os

from setuptools import Extension, find_namespace_packages, setup

setup(
    packages=find_namespace_packages(include=["jijbench*"]),
    ext_modules=[
        Extension(
            "jijbench.__marker__",
            [os.path.join("jijbench", "__marker__.c")],
        )
    ],
    package_data={
        "": ["*.json", "*.JSON"],
    },
    include_package_data=True,
)
