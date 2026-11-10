# 23 Architecture model

The architecture model, `architecture/model.yaml`, is the one machine-readable source of the structure of the library.
The architecture checker compares it with the code and fails the build when they disagree.

<a id="purpose"></a>

## 23.1 Purpose

The model states the C4 context, the containers, the layers, the components and the rules.
Agents and people change the structure only through a reviewed change to the model.

<a id="model-contents-and-matching"></a>

## 23.2 Model contents and matching

**Table 23.1: Model contents**

| Element | Content |
| --- | --- |
| System, people and external systems | The C4 context |
| Relationships | Who uses what, and with which technology |
| Containers | The deployable and runtime units |
| Layers | A directory, the layers it can use, its external modules with reasons, and whether its components must stay independent |
| Components | The packages that each component owns, the components that it depends on and its external modules |
| Test modules | The modules that any test file can import |
| Test support | The packages of shared fakes and conformance suites, which only tests import |
| Rules | Forbidden imports and symbols, their scope, their excepted components and a fix message |

Each package sits under exactly one layer directory, and the checker derives each component's layer from its packages.
A component names its packages and the containers that run it, and a pattern that ends in `/...` owns a whole subtree.
The model declares no package that does not exist, so components enter the model with their code.

The model uses these matching rules:

- A forbidden import, a forbidden symbol and an external entry each match the path itself.
- A pattern that ends in `/...` also matches every package below it.
- An external pattern never crosses into another module, so each nested module or `/vN` version needs its own entry.
- A match never extends a path segment, so `k8s.io/api` does not match `k8s.io/apiserver`.
- An import needs both the layer entry and the component grant.

The checker embeds the JSON Schema that defines the format.
The checker adds the semantic rules: layer uses and component dependencies have no cycle, and identifiers are unique.
No layer directory contains another, and each reference resolves.

<a id="rules"></a>

## 23.3 Rules

The layer uses and external lists already control imports of third-party modules and of other layers.
Rules cover standard-library packages and symbols, which external lists cannot govern.

**Table 23.2: Model rules**

| Rule | Forbids |
| --- | --- |
| `no-environment-access` | Environment reads and writes, including the home and temporary directories |
| `no-flag-package` | The `flag` package |
| `no-process-execution` | `os/exec` and process start functions |
| `explicit-logging` | The `log` package and the `slog` functions that set or read the default logger. sloglint forbids the other package-level `slog` functions. |
| `no-process-globals` | Root contexts, process exit and arguments, signal handling, and the default HTTP client, transport and multiplexer |

The model states the fix for each rule.
Rules that apply to all code stay in the golangci-lint configuration, such as the bans on `math/rand` and `reflect.DeepEqual` in tests.
No rule forbids `time.Now`, because `testing/synctest` fakes the clock inside each test bubble.

The model's `test_modules` list limits test imports, so tests cannot import an assertion library.

<a id="design-records"></a>

## 23.4 Design records

A design record is a file `architecture/designs/NNNN-slug.md` with its number, title and issue in its front matter.
Its body states the problem, the design and the alternatives.
It records reasons, not facts: git history of the model is the log of structural facts.

A structural change to the model adds a design record in the same pull request.
Every change to the model other than its descriptive text is structural.
The checker fails a pull request that changes the model's structure without a new record.

Design record 0001 creates the container, the layer and the rules.
The first implementation pull request adds the first components with their packages.

<a id="checks"></a>

## 23.5 Checks

**Table 23.3: Architecture checks**

| Check | Tool |
| --- | --- |
| Model format, semantics and the record rule | `make arch-check` locally, and `make ci-check` with the base branch in CI |
| Packages, imports, modules and rules | `make arch-check`, before any package |
| Rendered C4 view | The checker, when it supports rendering |

The checker is a public tool, `archcheck`, which several projects share.
The tools module in `tools/` pins it by commit until it publishes a release.
The checker holds only generic code: the model, the records and the code of this repository stay here.
