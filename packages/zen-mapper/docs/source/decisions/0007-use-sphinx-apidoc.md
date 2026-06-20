---
status: "accepted"
date: "2026-06-20"
---
# Use sphinx-apidoc

## Context and Problem Statement

We would like to automatically generate api docs using the doc strings of our
methods. What tool should we use to generate those?

## Decision Drivers

- Ease of use, the more automatic the tool is the better
- Well supported, we don't want the tool we use to suddenly go away
- Supports type annotations, we would like to minimize the duplication of type
  information if at all possible

## Considered Options

- [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)
- [sphinx autoapi](https://sphinx-autoapi.readthedocs.io)

## Decision Outcome

"sphinx-apidoc", because it is distributed with sphinx and supports type
annotations.

## Pros and Cons of the Options

### sphinx-apidoc

- Good, because it is distributed with sphinx
- Good, because it supports type annotations
- Bad, because it requires running a separate tool to generate the api docs
- Bad, because migration will take work

### sphinx autoapi

- Good, because we are already using it
- Bad, because we now rely on a third party
- Bad, because its type annotation support is "experimental" and may be removed
