.DEFAULT_GOAL := help
.PHONY: help tools hygiene vale-lint arch-check go-lint go-security go-test go-fuzz go-regression docs docs-serve prose-all ci-guard ci-lint ci-test ci-arch ci-check

MISE ?= mise
PYTHON = $(MISE) exec -- uv run --locked --no-build python
GO = $(MISE) exec -- go
ARCHCHECK = .cache/bin/archcheck
GOLANGCI = .cache/bin/golangci-lint
GOVULNCHECK = .cache/bin/govulncheck
COVERAGE = .cache/bin/go-test-coverage
REGRESSIONTEST = .cache/bin/regressiontest
# Local commands require explicit files, packages or one test.
export FILES PKGS TEST

help:
	@echo 'Local: make vale-lint FILES=docs/24-writing-guide.md'
	@echo 'Local: make go-lint PKGS="./internal/first ./internal/second"'
	@echo 'Local: make go-test PKGS=./internal/first TEST=TestName'
	@echo 'Local: make arch-check, make hygiene, make docs, make docs-serve'
	@echo 'CI only: make go-security, make go-fuzz, make go-regression'
	@echo 'Full checks: pull request CI only (make ci-check).'

tools:
	$(MISE) install
	$(MISE) exec -- uv sync --locked --no-build

hygiene:
	$(MISE) exec -- uv run --locked --no-build pre-commit run

# The guard runs before any full check. A generic CI=true does not enable these commands locally.
define require_ci
	@if test "$${GITHUB_ACTIONS:-}" = true && test -n "$${GITHUB_RUN_ID:-}"; then :; \
	else echo 'Full checks run only in CI. Use explicit FILES, PKGS or TEST locally.' >&2; exit 2; fi
endef

# Explicit Go package directories in PKGS select a local run. Without PKGS, the complete module
# runs, and only CI can start it. The shell variable full is empty for a selected run.
define select_go_packages
if test -n "$$PKGS"; then full=; \
	case "$$PKGS" in *'*'*|*'?'*|*'['*) echo 'List explicit packages instead of wildcard patterns.' >&2; exit 2;; esac; \
	for pkg in $$PKGS; do case "$$pkg" in *..*|*[!a-zA-Z0-9_./-]*) \
		echo 'PKGS must list package directories, not patterns or options.' >&2; exit 2;; ./*) ;; \
		*) echo 'Each package in PKGS must start with ./.' >&2; exit 2;; esac; \
	test -d "$$pkg" || { echo "No package directory: $$pkg" >&2; exit 2; }; full=selected; done; \
	test "$$full" = selected || { echo 'Set PKGS to explicit Go package directories.' >&2; exit 2; }; full=; \
elif test "$${GITHUB_ACTIONS:-}" = true && test -n "$${GITHUB_RUN_ID:-}"; then full=1; \
else echo 'Set PKGS to explicit Go package directories. The complete module runs only in CI.' >&2; exit 2; fi
endef

# The complete module has no Go package until the first library package arrives.
define go_packages_exist
test -n "$$($(GO) list ./... 2>/dev/null)"
endef

vale-lint:
	@test -n "$$FILES" || { echo 'Set FILES to explicit Markdown files.' >&2; exit 2; }
	@for file in $$FILES; do case "$$file" in *.md) test -f "$$file" || exit 2;; *) echo 'FILES must contain Markdown files.' >&2; exit 2;; esac; done
	$(MISE) exec -- vale --config .vale.ini $$FILES

# The tools module pins the Go tools, so the library module never depends on them.
$(ARCHCHECK): tools/go.mod tools/go.sum
	$(GO) build -modfile tools/go.mod -o $(ARCHCHECK) github.com/OpenRAE/go-rae-archcheck/cmd/archcheck

$(GOLANGCI): tools/go.mod tools/go.sum
	$(GO) build -modfile tools/go.mod -o $(GOLANGCI) github.com/golangci/golangci-lint/v2/cmd/golangci-lint

$(GOVULNCHECK): tools/go.mod tools/go.sum
	$(GO) build -modfile tools/go.mod -o $(GOVULNCHECK) golang.org/x/vuln/cmd/govulncheck

$(COVERAGE): tools/go.mod tools/go.sum
	$(GO) build -modfile tools/go.mod -o $(COVERAGE) github.com/vladopajic/go-test-coverage/v2

$(REGRESSIONTEST): tools/go.mod tools/go.sum
	$(GO) build -modfile tools/go.mod -o $(REGRESSIONTEST) github.com/OpenRAE/rae/implementations/tooling/go/cmd/regressiontest

arch-check: $(ARCHCHECK)
	$(ARCHCHECK)

# The Go lint gate of section 22.2. The complete run also checks the module requirements.
go-lint:
	@$(select_go_packages); set -e; \
	$(MAKE) --no-print-directory $(GOLANGCI); \
	if test -n "$$full"; then \
		$(GO) mod tidy -diff; (cd tools && $(GO) mod tidy -diff); \
		if ! $(go_packages_exist); then echo 'No Go package yet, so only go mod tidy runs.'; exit 0; fi; \
		pkgs=./...; \
	else pkgs=$$PKGS; fi; \
	$(GOLANGCI) run $$pkgs; $(GO) vet $$pkgs; $(GO) fix -diff $$pkgs

# The dependency checks of section 22.5 and the source gates of section 22.6. The license scan
# fails on restricted, forbidden and unknown licenses. The tools module never ships, so the scan
# leaves it out.
go-security:
	$(require_ci)
	$(MAKE) --no-print-directory $(GOVULNCHECK)
	$(GO) mod verify
	$(GO) mod download
	$(MISE) exec -- trivy fs --scanners license --severity UNKNOWN,HIGH,CRITICAL --exit-code 1 --quiet \
		--skip-dirs tools --skip-dirs .venv --skip-dirs .cache --skip-files uv.lock .
	@if $(go_packages_exist); then $(GOVULNCHECK) ./...; else echo 'No Go package yet, so govulncheck waits.'; fi

# The Go tests of section 22.4, with the race detector and shuffled order. The complete run also
# applies the coverage gate of at least 90% for each package. TEST selects one test of a local run.
go-test:
	@$(select_go_packages); set -e; \
	if test -n "$$full"; then \
		if ! $(go_packages_exist); then echo 'No Go package yet, so the tests wait.'; exit 0; fi; \
		mkdir -p .cache; $(MAKE) --no-print-directory $(COVERAGE); \
		$(GO) test -race -shuffle=on -covermode=atomic -coverprofile=.cache/coverage.out ./...; \
		$(COVERAGE) --config .testcoverage.yml; \
	else run=; if test -n "$$TEST"; then case "$$TEST" in Test*) ;; *) \
		echo 'TEST must name one Go test that starts with Test.' >&2; exit 2;; esac; \
		case "$$TEST" in *[!a-zA-Z0-9_]*) echo 'TEST must name one Go test.' >&2; exit 2;; esac; \
		run="-run=^$$TEST\$$"; fi; \
		$(GO) test -race -shuffle=on $$run $$PKGS; fi

# The fuzz targets of section 22.4. Go lists them, and each target runs for a short time. A failing
# input lands in testdata/fuzz, where go test then runs it on every change. CI keeps that input as
# an artifact.
go-fuzz:
	$(require_ci)
	@set -e; if ! $(go_packages_exist); then echo 'No Go package yet, so no fuzz target runs.'; exit 0; fi; \
	mkdir -p .cache; module="$$($(GO) list -m)"; \
	listed="$$($(GO) test -run='^$$' -list='^Fuzz' ./...)" || { printf '%s\n' "$$listed" >&2; exit 1; }; \
	printf '%s\n' "$$listed" | awk -v module="$$module" '/^Fuzz/ { names[++n] = $$0; next } \
		$$1 == "ok" { if (index($$2, module) == 1) $$2 = "." substr($$2, length(module) + 1); \
		for (i = 1; i <= n; i++) print $$2 ":" names[i]; n = 0 }' | sort > .cache/fuzz-targets; \
	test -s .cache/fuzz-targets || { echo 'No fuzz target yet.'; exit 0; }; \
	while IFS=: read -r pkg name; do \
		echo "go test -fuzz $$name $$pkg"; \
		$(GO) test -run='^$$' -fuzz="^$$name\$$" -fuzztime=15s "$$pkg" || exit 1; \
	done < .cache/fuzz-targets

# The regression proof of section 22.4: each new TestRegression test in a *_regression_test.go
# file must fail on the base revision with a test failure. CI sets BASE_REF for pull requests.
go-regression:
	$(require_ci)
	$(MAKE) --no-print-directory $(REGRESSIONTEST)
	$(MISE) exec -- $(CURDIR)/$(REGRESSIONTEST) -base "origin/$${BASE_REF:-dev}"

docs:
	$(PYTHON) -m mkdocs build --strict

docs-serve:
	$(PYTHON) -m mkdocs serve

# The Vale rules of section 24.5 check every tracked Markdown file.
prose-all: ci-guard
	$(MISE) exec -- vale --config .vale.ini $$(git ls-files '*.md')

ci-guard:
	$(require_ci)

# The CI entry points clear PKGS and TEST, so the Go gates always check the complete module.
ci-lint: ci-guard
	$(MAKE) go-lint go-security PKGS= TEST=

ci-test: ci-guard
	$(MISE) exec -- vale --version
	$(MAKE) go-test go-fuzz go-regression PKGS= TEST=

# CI also compares the model with the base branch, so a structural change needs a new design record.
ci-arch: ci-guard
	$(MAKE) --no-print-directory $(ARCHCHECK)
	@if [ -n "$$BASE_REF" ] && git cat-file -e "origin/$$BASE_REF:architecture/model.yaml" 2>/dev/null; then \
		git show "origin/$$BASE_REF:architecture/model.yaml" > .cache/base-model.yaml; \
		git ls-tree --name-only "origin/$$BASE_REF" architecture/designs/ > .cache/base-designs.txt; \
		$(ARCHCHECK) -base-model .cache/base-model.yaml -base-designs .cache/base-designs.txt; \
	else $(ARCHCHECK); fi

ci-check: ci-guard
	$(MAKE) ci-lint
	$(MAKE) ci-test
	$(MAKE) ci-arch docs prose-all
