# 22 Engineering standards

These standards apply from the first line of code.
Tools enforce every rule that a tool can check, with no baseline.
Review covers the rest, and this chapter names which rules are review rules.

<a id="scope"></a>

## 22.1 Scope

The standards cover the Go code and the documents.
Generated code is checked through its generator.
Vendored code keeps its upstream conventions.

<a id="go-style-and-structure"></a>

## 22.2 Go style and structure

The [Google Go Style Guide](https://google.github.io/styleguide/go/) is the style.
Its guide and decisions are normative, and its best practices are advisory.
A published contract wins over a style rule.

**Table 22.1: Go rules and their checks**

| Rule | Check |
| --- | --- |
| Formatting | `gofmt` and `goimports` |
| Names: MixedCaps, consistent initialisms, short receivers | staticcheck with `checks: ["all"]`, and revive with its default rules enabled |
| Package names state one responsibility, never `util`, `common` or `helper` | revive `package-naming` with a configured list of bad names, and the architecture model |
| Every exported name and package has a doc comment | revive `exported` and `package-comments`, staticcheck ST1000 and ST1020 to ST1022, with no `comments` exclusion preset |
| Errors are handled and compared with `errors.Is` or `errors.As`. `%w` wraps only where callers can inspect the cause. | errcheck, errorlint with its `errorf` check off, and review |
| Error strings are lower case with no final punctuation | staticcheck ST1005 |
| `context.Context` is the first parameter and is never stored in a struct | revive `context-as-argument`, containedctx, contextcheck |
| Logging uses an injected `*slog.Logger` with constant messages and the `Context` methods | sloglint with `static-msg`, `no-global: all` and `context: scope` |
| No `init` functions and no mutable package variables | gochecknoinits, gochecknoglobals |
| Switches over enumerations are exhaustive | exhaustive |
| Interfaces stay small, and functions return concrete types | iface, interfacebloat, ireturn |
| Cognitive complexity is 15 or less | gocognit |
| HTTP requests carry a context, and response bodies close | noctx, bodyclose |
| Tests print got before want | Review |
| `%w` sits at the end of an error format, or at the start for a sentinel error | Review |
| Function names have no `Get` prefix, unless the concept is a get | Review |
| Test helpers call `t.Helper` | thelper |
| A suppression names one linter and says why the finding is false | nolintlint and review |
| Clarity, simplicity, least mechanism and local consistency | Review |

golangci-lint v2 runs at a pinned version, with the configuration of this repository, which enables every linter except a listed set.
Each disabled linter has a reason from the Google guide in the configuration.
lll, nonamedreturns, exhaustruct, testpackage, err113, noinlineerr, wsl, wsl_v5, nlreturn, varnamelen and paralleltest stay off, because the guide permits what they forbid.

For example, the guide endorses short names such as `r` and `w`, and it does not require `t.Parallel`.

depguard stays off, because the architecture checker owns imports.
wrapcheck stays off, because the guide adds context to an error only where it helps the reader.
CI also runs `go vet`, `go mod tidy -diff` and `go fix -diff`.

The module has `internal/` for every package that is not public, as the [Go module layout guidance](https://go.dev/doc/modules/layout) recommends.
A public package declares its layer in the architecture model before its first file.

Constructors build the object graph by hand, with no injection framework and no registration in `init`.
The toolchain is the latest stable Go release, and `go.mod` pins both the language version and the toolchain.

<a id="contract-based-go"></a>

## 22.3 Contract-based Go

**Table 22.2: Contract-based idioms**

| Idiom | Rule |
| --- | --- |
| Validated types | A type with invariants keeps its fields unexported. Its constructor checks the invariants and returns an error. |
| One boundary check | Input from outside the process is checked once where it enters. Validated types then carry it. |
| Consumer interfaces | A consumer declares the small interface that it needs. An interface with several implementations has an exported conformance test suite, as `testing/fstest.TestFS` does. |
| Errors for bad input | A function returns an error for bad input. It panics only when an internal invariant is broken. |
| Strict decoding | JSON input decodes with `encoding/json/v2` and `json.RejectUnknownMembers(true)`, which also rejects duplicate names and trailing data. |
| Contained files | File access inside a directory uses `os.Root`. |
| Custom code | Code that a standard library, a battle-tested library or an established standard covers needs a design record with the exceptional reason. |

<a id="tests"></a>

## 22.4 Tests

Tests use the standard `testing` package and `go-cmp`, with no assertion library.
Failure messages print the actual value before the wanted value.
Table tests, golden files and `t.Helper` are the normal forms, and a golden file change is a contract change.

**Table 22.3: Test sizes**

| Size | Limits | Gate |
| --- | --- | --- |
| Small | One process, no network, no disk, no sleep | Every pull request |
| Medium | Processes and network on `localhost` only | Every pull request |
| Large | Several machines or a cluster | Merge to `dev` and on schedule |

The sizes follow [Software Engineering at Google](https://abseil.io/resources/swe-book/html/ch11.html).
A medium test skips under `-short` and fails in CI when its service is missing, so no build tag hides it from tools.
Shared fakes and conformance suites live in the test support packages that the model lists, and only tests import them.

- CI runs `go test -race -shuffle=on`.
- Code that waits or retries uses `testing/synctest`.
- Tests replace adapters with fakes through consumer interfaces and never patch package state.
- Every parser of untrusted input and every canonical encoding has a fuzz target or a reviewed applicability decision.
- CI runs each fuzz target for a short time, and a failing input stays in `testdata/fuzz`.

go-test-coverage keeps statement coverage at or above 90% for each package.
Mutation testing is required for new code.
CI gains a mutation gate when the repository pins a mutation tool.

A regression test uses the `TestRegression` prefix in a `*_regression_test.go` file.
The `regressiontest` tool runs the head's tests on the base revision and accepts only a test failure, never a build failure.

<a id="dependencies"></a>

## 22.5 Dependencies

Code prefers the standard library, then `golang.org/x` modules, then established modules.
An established module is maintained, has a permitted license and passes `govulncheck`.
The architecture model lists each permitted module for each layer, with a reason.

Updates arrive in grouped pull requests after a delay, never on the day of release.
Dependabot opens one pull request for each ecosystem each week, after a cooldown of one day.
It has no mise ecosystem, so the tools that mise pins update by hand.

CI runs `go mod verify`, and a Trivy license scan of the module fails on restricted, forbidden and unknown licenses.
The tools module never ships, so the scan leaves it out.

Go tools live in a separate tools module or come from mise, never from the library `go.mod`.

<a id="security-gates"></a>

## 22.6 Security gates

**Table 22.4: Security gates**

| Gate | Tool |
| --- | --- |
| Known vulnerabilities in Go code | `govulncheck` |
| Code scanning | CodeQL for Go, and gosec in golangci-lint |
| Secrets in the repository | gitleaks in commit hooks and CI |
| Workflow supply chain | Actions pinned by full commit SHA, least-privilege tokens, actionlint, zizmor and OpenSSF Scorecard |

<a id="design-step-and-review"></a>

## 22.7 Design step and review

Each piece of work updates its design first: the affected documents and the model.
A structural change, such as a new component, component dependency, external module, layer or rule, adds a design record in the same pull request.
[Architecture model](23-architecture-model.md) defines the records.

Before each merge, the author runs an adversarial review and a consistency check of the change against the whole design.
The pull request records each finding and how it was fixed or why it was not.
The person who merges reads that record.

Review follows Google's [code review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html): design, functionality, complexity, tests, naming, comments, style, consistency and documentation.
The reviewer reads every line and checks that the change improves code health.
Review rules from table 22.1 are part of that review.

<a id="branches-commits-and-checks"></a>

## 22.8 Branches, commits and checks

Branches are short and merge into `dev` often, as [trunk-based development](https://dora.dev/capabilities/trunk-based-development/) recommends.
Promotions from `dev` to `main` use merge commits.

Pull request titles use Conventional Commits, and commits are signed.
Local commands run targeted checks with explicit targets, and the complete suite runs in CI.
A person merges every pull request after all gates pass.

The required `Validate` result includes the repository checks and CodeQL.
CI calls the CodeQL workflow through `workflow_call`, and the weekly schedule calls the same analysis jobs.
Failed, cancelled, missing or unexpected skipped results fail validation, and only an empty language list permits a skipped analysis.

`make go-lint` and `make go-test` need explicit Go package directories in `PKGS` locally.
`make go-test` also accepts one Go test name in `TEST`.
Without `PKGS`, these two targets check the complete module, and only CI can start that run.

The gates `go-security`, `go-fuzz` and `go-regression` run only in CI.
Each guarded command checks for CI before it builds or starts a tool.
