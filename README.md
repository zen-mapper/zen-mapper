# Zen Mapper

> Perfection is achieved, not when there is nothing more to add, but when there
> is nothing left to take away.
>
> Antoine de Saint-Exupéry

Zen mapper is a minimal implementation of the TDA algorithm Mapper

# Installation

There are two supported methods of installation

## PyPi

Zen Mapper has a distribution on [PyPi](https://pypi.org/project/zen-mapper/).
You can install it using pip:
```sh
pip install zen-mapper
```

## Nix

Zen Mapper is also packaged as a nix flake. If you have nix installed and flake
support enabled. You can create a dev environment with Zen Mapper installed
easily enough:
```sh
nix flake new -t github:zen-mapper/zen-mapper new_project
cd new_project
nix develop
```
Will drop you into a shell with python 3.11 and Zen Mapper configured.
