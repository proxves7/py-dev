# py-dev

## uv
```
# install uv
$ pip3 install uv

# list python version
$ echo 'export PATH=$HOME/Library/Python/3.9/bin:$PATH' >> ~/.bash_profile
$ uv python list

# install python
$ uv python install 3.13

# remove python
$ uv uninstall install 3.13
```

## uv project
```
# init project
$ uv init --python 3.13

# create venv
$ uv venv --python 3.13

# install package
$ uv add torch

# remove package
$ uv remove torch

# sync project
$ uv sync

# run t1_torch.py
$ uv run t1_torch.py
```