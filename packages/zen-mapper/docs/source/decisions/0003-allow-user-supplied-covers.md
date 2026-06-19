---
status: "accepted"
date: "2026-06-21"
---
# Allow user supplied covers

## Context and Problem Statement

The mapper algorithm needs a cover of the high dimensional data in order to
function. Should the end user be able to define their own covers?


## Decision Drivers

- Flexibility, zen mapper should enable as many use cases as possible
- Extensibility, zen mapper should be open to extensions

## Considered Options

- Enable user supplied covers
- Hardcode the most common covers

## Decision Outcome

"Enable user supplied covers", because this is the only thing which facilitates
extensibility.
