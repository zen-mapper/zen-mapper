---
status: "accepted"
date: "2026-06-21"
---
# Encode covers as iterables

## Context and Problem Statement

The mapper algorithm needs a cover of the high dimensional data in order to
function. The end user should be able to supply this. There are two orthogonal
questions that need answering.

1. Should the user provide the raw cover or a method for producing a cover
1. How should the data of a cover be encoded

This adr seeks to address the second question. We will take it as a given for
now that we represent cover elements as numpy arrays of indices into the
original data. This can be examined in the future but it is how we have
implemented thing thus far.

## Decision Drivers

- Minimal overhead for people implementing new covers
- Flexibility, as many use cases as possible should be enabled
- Extensibility, in the future we should be able to add functionality without
  breaking old implementations
- Type safety, python type checkers should be able to tell what is going on

## Considered Options

- An iterable of cover elements
- An iterable of id, cover element pairs
- A mapping of cover elements
- Some custom base class

## Decision Outcome

"An iterable of cover elements", because it is the simplest of the proposed
options to implement and does not seem to hamper us in anyway.

## Pros and Cons of the Options

### An iterable of cover elements

- Good, because it requires very little of the author
- Good, because it gives us everything we actually need
- Good, because it trivially enables streaming covers
- Bad, because it does not support random access
- Bad, because it forces the cover elements to be keyed by integers

### An iterable of id, cover element pairs

- Good, because it gives us everything we actually need
- Good, because it trivially enables streaming covers
- Good, because it enables the cover elements to be keyed by arbitrary elements
- Bad, because implementing this interface will require boilerplate in most
  cases just to enable features in rare cases
- Bad, because it does not support random access

### A mapping of cover elements

- Good, because it allows arbitrary keying of cover elements
- Bad, because it requires more work on the implementer, there are not that
  many built in map types
- Bad, because it complicates streaming covers. There are no built in lazy map
  types.

### Some custom base class

- Good, because it enables us to enforce our contract strictly
- Bad, because it requires the implementer to extend our class instead of
  reusing types that already exist

## More Information

- [ADR-0003](./0003-allow-user-supplied-covers.md) reasons on allowing user supplied covers
- [ADR-0004](./0004-use-an-explicit-cover.md) reasons to supply a cover instead of a scheme
- [ADR-0005](./0005-use-numpy-array-to-encode-cover-elements.md) reasons to
  encode cover elements as numpy arrays
