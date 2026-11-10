# gotstate

[![CI](https://github.com/Brad-Edwards/gotstate/actions/workflows/ci.yml/badge.svg)](https://github.com/Brad-Edwards/gotstate/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Brad-Edwards_gotstate&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Brad-Edwards_gotstate)

gotstate is a Go library for hierarchical finite state machines.
This repository holds the engineering baseline and no library code yet.

## Earlier releases

The tags `v1.0.0` to `v1.0.4` hold an earlier Python library.
Go releases of this module follow those tags.

## Read the documentation

The documents start at `docs/00-contents.md`.
Run `make docs-serve` to read them in the local reader, or `make docs` to build them.

## Repository layout

| Path | Content |
| --- | --- |
| `docs/` | The numbered documents: engineering standards, architecture model guide and writing guide |
| `architecture/` | The architecture model and its design records |
| `go.mod` | The Go module of the library |
| `tools/` | The Go tools module, which pins the Go tools of the checks |
| `styles/` | The Vale rules for ASD-STE100 writing |

## Set up and check

Install [mise](https://mise.jdx.dev/getting-started.html) 2026.10.0 or newer, then run `make tools`.
Local commands run targeted checks with explicit targets, for example `make vale-lint FILES=docs/24-writing-guide.md`.
Run `make arch-check` to check the model and the code with the architecture checker.

Go checks need explicit packages, for example `make go-test PKGS=./internal/example`.
Pull request CI runs `make ci-check`, and full checks stop outside CI.

## Contribute

[`CONTRIBUTING.md`](CONTRIBUTING.md) describes the setup, the checks and the standards.
Report a vulnerability as [`SECURITY.md`](SECURITY.md) states, and follow the [code of conduct](CODE_OF_CONDUCT.md).

## License

MIT, as `LICENSE` states.
