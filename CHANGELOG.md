# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- overlay for nixpkgs
- template flake

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
