# Use MADR

## Context and Problem Statement

We wish to record important decisions made in this project. How should we do
that?

## Decision Drivers

* Allow new contributors to easily understand past decisions
* Avoid vendor/tool lock in
* Minimize developer overhead

## Considered Options

* [MADR 4.0.0](https://adr.github.io/madr/)
* [Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions.html)
  template
* Github issues
* Project-wide design document
* No specified format

## Decision Outcome

Chosen option: "[MADR 4.0.0](https://adr.github.io/madr/)" because

- A project-wide design document sounds tiring
- No specified format leads to blank page syndrome
- Github issues locks us into a specific vendor
- Markdown is lightweight and easy to write
- Using a standard opens us up to the potential for tool re-use
- Just needed to choose *something*
