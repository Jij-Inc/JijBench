import os, sys

from setuptools import find_namespace_packages, setup

setup_requires = [
    "setuptools_scm[toml]",
]

if any(arg in sys.argv for arg in ("pytest", "test")):
    setup_requires.append("pytest-runner")

try:
    from Cython.Compiler import Options
    from Cython.Distutils.build_ext import build_ext as build_ext

    Options.embed_pos_in_docstring = True
    Options.cimport_from_pyx = True
    Options.annotate = True

    def get_export_symbols_fixed(self, ext):
        names = ext.name.split(".")
        if names[-1] != "__init__":
            initfunc_name = "PyInit_" + names[-1]
        else:
            # take name of the package if it is an __init__-file
            initfunc_name = "PyInit_" + names[-2]
        if initfunc_name not in ext.export_symbols:
            ext.export_symbols.append(initfunc_name)
        return ext.export_symbols

    if os.name == "nt":
        build_ext.get_export_symbols = get_export_symbols_fixed
except ImportError:
    from setuptools.command.build_ext import build_ext as build_ext

    setup_requires.append("Cython")


def cython_extension():

    try:
        from Cython.Distutils.extension import Extension

        if os.name == "nt":
            extra_compile_args_cfg = ["/EHsc"]
        else:
            extra_compile_args_cfg = []

        extensions = [
            Extension(
                "jijbench.__marker__",
                [os.path.join("jijbench", "__marker__.py")],
                extra_compile_args=extra_compile_args_cfg,
                define_macros=[("CYTHON_TRACE", "1")],
            )
        ]

    except ImportError:
        from setuptools import Extension

    compiler_directives_cfg = {
        "language_level": 3,
        "embedsignature": True,
        "binding": True,
        "profile": True,
        "linetrace": True,
        "emit_code_comments": True,
        "remove_unreachable": False,
    }

    try:
        from Cython.Build import cythonize

        cy_ext_module = cythonize(
            extensions,
            compiler_directives=compiler_directives_cfg,
            annotate=True,
        )
    except ImportError:

        cy_ext_module = list()
    return cy_ext_module


setup(
    setup_requires=setup_requires,
    install_requires=[
        "openjij ~= 0.5.8",
        "jijzept ~= 1.10.9",
        "jijmodeling ~= 0.9.25",
        "numpy < 1.23.0",
        "pandas ~= 1.4.3",
        "matplotlib ~= 3.5.2",
        'pyqubo<1.1.0; python_version < "3.10"',
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=cython_extension(),
    packages=find_namespace_packages(include=["jijbench*"]),
    package_data={
        "": ["*.json", "*.JSON"],
    },
    include_package_data=True,
)
