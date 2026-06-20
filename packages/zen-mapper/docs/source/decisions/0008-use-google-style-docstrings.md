---
status: "accepted"
date: "2026-06-20"
---
# Use google style docstrings

## Context and Problem Statement

We would like to automatically generate api docs using the docstrings of our
methods. How should we format these docstrings?

## Decision Drivers

- Ease of writing, it should be easy for an author to write these doc strings
- Ease of reading, it should be easy to read these docstrings without rendering them
- Supported by sphinx
- Supports untyped parameters and returns, we have type hints for these

## Considered Options

- Google style docstrings
- Numpy style docstrings
- Sphinx style docstrings


## Decision Outcome

"Google style docstrings", because it supports untyped returns, is common, is
easy to read, and easy to write.

## Pros and Cons of the Options

### Google style docstrings

- Good, because common
- Good, because supported in sphinx via napoleon
- Good, because it supports untyped return values
- Bad, because we will have to convert our current docstrings over

### Numpy style docstrings

- Good, because common
- Good, because supported in sphinx via napoleon
- Good, because we already use it
- Bad, because does not support untyped return values

### Sphinx style docstrings

- Good, because supported in sphinx natively
- Good, because it supports untyped return values
- Bad, because we will have to convert our current docstrings over
- Bad, because I am not that familiar with them
