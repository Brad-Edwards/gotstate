#!/usr/bin/env bash
set -euo pipefail

# Run pytest with coverage
coverage run --branch --source=hsm -m pytest

# Generate coverage report
coverage report --fail-under=90

# Generate HTML report for detailed view
coverage html