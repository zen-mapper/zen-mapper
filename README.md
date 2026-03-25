# Zen Mapper Monorepo

Home of Zen + Kaiju Mapper. Zen mapper is a minimal implementation of the TDA
algorithm Mapper while Kaiju mapper is a collection of tools built upon Zen
mapper.

Each lives in their own directory under `packages`.

Documentation for zen-mapper can be found [here](https://zen-mapper.github.io/zen-mapper/).

Documentation for kaiju-mapper is not actively hosted anywhere right now.

## Status

Both packages should be considered beta software at the moment.

## Development

To enter a dev shell with all needed dependencies there are roughly two options.

If you have [nix](https://nixos.org) installed (and flakes enabled):

```sh
nix develop
```

If you have [uv](https://docs.astral.sh/uv/) installed, this will get you pretty close:

```sh
uv sync --group docs --group dev
source .venv/bin/activate
```
