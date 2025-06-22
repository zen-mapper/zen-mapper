# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- There is now a `Data_Balanced_Cover` which covers a one-dimensional data set
  with even bins.

### Change

- The clusterer protocol requires passing in the global dataset along with the
  indices to cluster now.
- The sk_learn adapter now accepts `ArrayLike` objects, not just `ndarray`s
- The clusterer protocol now allows returning `ArrayLike` objects, not just `ndarray`s

### Fixed

- Fixed a bug where empty nodes could slip into the mapper graph
- Fixed a bug where mapper tried to cluster empty cover elements

## [0.3.0] - 2025-06-02

### Added

- Komplex now has a `__len__` method which reports how many simplices are in
  the complex.
- Mapper now takes a `min_intersection` parameter to allow specifying the
  overlap in the nerve.
- `zen_mapper.cover.precomputed_cover` was added which allows specifying a
  precomputed cover instead of a cover scheme.

### Change

- Reorganized the entire package to allow for less painful imports. Everything
  is now accesible from the top level module `zen_mapper`
- The cover protocol previously required being an iterable of numpy arrays.
  This was loosened to allow for iterables of anything which is convertible to
  a numpy array.
- The clusterer protocol requires returning a metadata object now.

### Fixed

- `sk_learn` cluster adapter no longer panics if given a dataset with 1 element

## [0.2.0] - 2024-12-28

### Added

- The `n_elements` parameter for `Width_Balanced_Cover` now allows for a
  different number of `n_elements` to be set per axis.

## [0.1.5] - 2024-10-24

### Change

- Added support for python 3.10
- Mark python 3.10 as the minimum version.

## [0.1.4] - 2024-02-17

### Added

- overlay for nixpkgs
- template flake

### Fixed

- `width_balanced_cover` no longer panics if it is given integer data

### Change

- Mark python 3.11 as the minimum version. It already was the minimum version, it was just mislabeled.

### Removed

- Removed unused `.python-version` file

### Fixed

- Removed spurious logging about looking for simplices

## [0.1.3] - 2023-12-05

### Fixed

- sklearn adapter no longer panics if it has empty clusters

## [0.1.2] - 2023-10-23

### Added

- `vertices` parameter added to `simplex` and `komplex`
- `to_networkx` adapter for `komplex`

### Changed

- `simplex` now extends `tuple[int,...]`
