# Contribute to gotstate

gotstate is a Go library for hierarchical finite state machines.
Anyone can open an issue. The repository accepts pull requests from vetted authors only.

A contribution must keep the library tested and documented, and it must follow the standards below.

## Choose the route

- For a small documentation or test fix, open a pull request.
- For a change of behavior or of the public interface, open an issue first.
- Keep unrelated work in separate pull requests.
- Base each pull request on `dev`. The `main` branch holds the releases.

## Set up the repository

1. Install [mise](https://mise.jdx.dev/getting-started.html) 2026.10.0 or newer.
2. Run `make tools`.
3. Run `mise exec -- uv run --locked --no-build pre-commit install` to install the hooks.

## Check a change

`make ci-check` runs every check that CI runs, and it runs only in CI.
Locally, run the targeted checks that `make help` lists, with explicit `PKGS` or `FILES` values.
`make hygiene` runs the file hygiene, workflow and secret hooks on the staged files.

## Follow the standards

- Write Go in the Google Go style. golangci-lint enforces it.
- Keep statement coverage at or above 90% for each package.
- Give each new parser of untrusted input a fuzz target.
- Name a regression test `TestRegression...` in a `*_regression_test.go` file. It must fail on the base revision.
- Add a design record in `architecture/designs` for each structural change of `architecture/model.yaml`.
- Use a standard or widely used library before custom code, and record the reason for custom code.
- Write documentation in ASD-STE100 Simplified Technical English. Vale checks it.
- Write each pull request title as a Conventional Commit, for example `fix: report a missing state`.

[Chapter 22](docs/22-engineering-standards.md) of the documentation states these standards in full.

## Open a pull request

Fill in each section of the pull request template.
CI must pass before a merge.
The maintainer merges each pull request.
