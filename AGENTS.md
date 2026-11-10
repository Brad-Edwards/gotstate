# Repository instructions

This repository holds gotstate: a Go library for hierarchical finite state machines.

## Reuse and review

- Use the standard library, a battle-tested library or an established standard before custom code.
- Before custom code, record the standard alternative and the exceptional reason in a design record.
- Update the design first: the affected documents and the model.
- Before merge, run an adversarial review and a consistency check against the whole design.
- Record each finding and its disposition in the pull request.

[Section 22.7](docs/22-engineering-standards.md#design-step-and-review) of the engineering standards defines these rules.

## Governing documents

- `docs/00-contents.md` indexes the documents.
- `docs/22-engineering-standards.md` defines the coding, test and review rules.
- `docs/23-architecture-model.md` defines the architecture model and the design records.
- `docs/24-writing-guide.md` defines the ASD-STE100 writing rules.

## Checks and branches

Run only targeted checks locally, with explicit `FILES` or `PKGS` values.
Pull request CI runs `make ci-check`, which must pass before merge.
Target pull requests at `dev`.

## Workflow configuration

The `.ground-control.yaml` file holds the repository settings of the agent workflow.
