# JijBench

[![Test](https://github.com/Jij-Inc/JijBenchmark/actions/workflows/python-test.yml/badge.svg)](https://github.com/Jij-Inc/JijBenchmark/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/Jij-Inc/JijBenchmark/branch/main/graph/badge.svg?token=55341HSOIB)](https://codecov.io/gh/Jij-Inc/JijBenchmark)

## Coverage Graph

| **Sunburst**                                                                                                                                                                   | **Grid**                                                                                                                                                                   | **Icicle**                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <a href="https://codecov.io/gh/Jij-Inc/JijBenchmark"><img src="https://codecov.io/gh/Jij-Inc/JijBenchmark/branch/main/graphs/sunburst.svg?token=55341HSOIB" width="100%"/></a> | <a href="https://codecov.io/gh/Jij-Inc/JijBenchmark"><img src="https://codecov.io/gh/Jij-Inc/JijBenchmark/branch/main/graphs/tree.svg?token=55341HSOIB" width="100%"/></a> | <a href="https://codecov.io/gh/Jij-Inc/JijBenchmark"><img src="https://codecov.io/gh/Jij-Inc/JijBenchmark/branch/main/graphs/icicle.svg?token=55341HSOIB" width="100%"/></a> |

# How to use

## For Contributor

Use `pre-commit` for auto chech before git commit.
`.pre-commit-config.yaml`

```
# pipx install pre-commit 
# or 
# pip install pre-commit
pre-commit install
```

### Local Install

1. Load up a new Python [`venv`](https://docs.python.org/3.9/library/venv.html)

```sh
# Create a new virtual environment.
python -m venv .venv --upgrade-deps
```

```sh
# Activate the environment.
source .venv/bin/activate
```

```sh
# Alternatively you could use this command, for activate the environment.
. .venv/bin/activate
```

https://www.gnu.org/savannah-checkouts/gnu/bash/manual/bash.html#Bourne-Shell-Builtins

2. Install Dependency

Use [`pip-tools`](https://github.com/jazzband/pip-tools).

```sh
pip install pip-tools
pip-compile setup.cfg
pip-compile build-requirements.in 
pip-compile test-requirements.in 
pip-compile dev-requirements.in 
pip-sync requirements.txt build-requirements.txt dev-requirements.txt test-requirements.txt 
```

4. Install the module

```sh
python -m pip install .
```

https://pip.pypa.io/en/stable/cli/pip_install/\
OR

```sh
python setup.py install
```

https://docs.python.org/3.9/distutils/introduction.html

### Test Python

This test runs with [pytest](https://docs.pytest.org/en/7.1.x/) and [pytest-runner](https://github.com/pytest-dev/pytest-runner/)

```sh
pip install pip-tools
pip-compile setup.cfg
pip-compile build-requirements.in 
pip-compile test-requirements.in 
pip-sync requirements.txt build-requirements.txt test-requirements.txt 
python setup.py test
```

### Lint & Format Python

```sh
pip install pip-tools
pip-compile setup.cfg
pip-compile build-requirements.in 
pip-compile test-requirements.in 
pip-compile dev-requirements.in 
pip-compile format-requirement.in
pip-sync requirements.txt build-requirements.txt test-requirements.txt dev-requirements.txt format-requirements.txt 
```

Format

```
# jijcloudsolverapi
python -m isort --force-single-line-imports --verbose ./jijbench
python -m autoflake --in-place --recursive --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables ./jijbench
python -m autopep8 --in-place --aggressive --aggressive  --recursive ./jijbench
python -m isort ./jijbench
python -m black ./jijbench
# tests-python
python -m isort --force-single-line-imports --verbose ./tests
python -m autoflake --in-place --recursive --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables ./tests
python -m autopep8 --in-place --aggressive --aggressive  --recursive ./tests
python -m isort ./tests
python -m black ./tests
```

Lint

```
python -m flake8
python -m mypy jijbench
python -m pyright jijbench
python -m lizard --verbose -l python jijbench
```

You could use Pylight.
https://github.com/microsoft/pyright