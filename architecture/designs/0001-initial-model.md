---
design: 1
issue: 101
title: Initial architecture model for the Go library
---

# Design 0001: Initial architecture model for the Go library

## Problem

The Go library has no code yet.
Its dependency rules must exist before the first package, so that each package starts inside the rules.

## Design

The model declares the C4 context, one container, one layer and five rules.
The context has one person, the developer who imports the library, and no external system.
The container is the Go module.

The `internal` layer maps to the `internal` directory.
It uses no other layer and no external module, so each new import needs a reviewed change to the model.

The five rules keep environment reads, command-line flags, process execution, global logging and process globals out of library code.
The calling program owns all of them.

The model declares no component and no public package layer.
A component or a public package enters the model with its code, in the same pull request and with its own design record.
The test modules are `go-cmp` and `go-internal`, because tests use the standard `testing` package with those two modules.

## Alternatives

- Declare no layer. The model format needs at least one layer.
- Declare the target components now. The model then fails its first code check, because no package exists.
