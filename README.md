![tests](https://github.com/konradmalik/mlfix/actions/workflows/test.yaml/badge.svg) ![publishing](https://github.com/konradmalik/mlfix/actions/workflows/publish.yaml/badge.svg) [![PyPI version](https://badge.fury.io/py/mlfix.svg)](https://badge.fury.io/py/mlfix)

# mlfix

## Motivation

Currently the is no way to natively make models trained using mlflow portable across local machines or from remote to local,
while keeping all the benefits (working commands etc.) of the mlflow environment (see for example [this issue](https://github.com/mlflow/mlflow/issues/3144)).

This is especially bad in small teams, local development or just prototyping.

This tool makes it easy to fix existing mlflow artifact store to current path.

Future work may include migrating existing artifact stores, only specific experiments etc.

Currently [mlf-core](https://github.com/mlf-core/mlf-core) also supports such functionality, but if you are not using mlf-core and want just to fix your mlruns, this tiny tool will help you.

## Installation

This is tested for Python 3.6 to 3.9.

From [PyPI](https://pypi.org/project/mlfix/):

```bash
$ pip install mlfix
```

From the source code (in the main directory):

```bash
$ python -m pip install .
```

## Usage

```bash
$ mlfix path_to_artifact_store
```

That is it!

You must specify the name of the `mlruns` folder if it was different than the default in the former location of the store:

```bash
$ mlfix --mlruns-name nonstandard_mlruns path_to_artifact_store
```
