# General purpose functions

[Tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) python packaging.

The command `python3.13 -m build` generates two packages: a .tar.gz containing all the files in the package folder and a .whl (wheel) containing only the source code and files needed by pip to install the package.

The flag `--sdist` generates only .tar.gz

The flag `--wheel` generates only -whl

```
python3.13 -m build --wheel
```

The generated package can be installed/uninstalled with `pip`. `Pip` automatically compiles the modules into .pyc files when installing.

```
python3.13 -m pip install --force-reinstall --no-dependencies --break-system-packages ./dist/fnc-0.0.0-py3-none-any.whl

python3.13 -m pip uninstall --break-system-packages fnc
```

Download dependencies with `pip`:

With the flag `--no-deps` it does not download all the dependencies of the requested package.

With the flag `--no-binary=:all:` it downloads only the .tar.gz, and without the .whl

```
python3.13 -m pip download pygaia==3.0.3 -d ./src/dep --no-deps
```

Check python code with `pylint` (`--disable={C,R,W,E}` where: C=Comment/Style, R=Recommendations, W=Warnings, E=Errors)

```
pylint -j 4 --output-format=colorized --max-line-length 120 --load-plugins=pylint.extensions.docparams,pylint.extensions.docstyle
```

Pylint for Jupyter notebooks:

```
nbqa pylint --output-format=colorized --disable={C0301,C0103,C0114,C0116,R0801,W0621} *.ipynb
```

Check import times with `tuna`:

```
python3.13 -X importtime -c "import fnc" 2> /tmp/fnc.log ; tuna /tmp/fnc.log
```

Profile execution of a python script `p.py` with `cProfile` and `snakeviz`:

```
python3.13 -m cProfile -o /tmp/p.log p.py
snakeviz /tmp/p.log
```

Run `jupyter-lab`:

```
jupyter-lab --browser=chromium
```

Run test:

```
python3.13 -m pytest -Wall -vs --durations=0 tests/tests.py
```

with options:

- `-Wall` : show all warnings

- `-x` : stops after first failure

- `-v` : verbose output

- `-s` : prints output from print()

- `--durations=0` : Print elapsed time for each test


- `-k` 'name_test_1 and name_test_2' : Run name_test_1 and name_test_2 only.

- `-k` 'not name_test_1 and not name_test_2' : Exclude name_test_1 and name_test_2.
