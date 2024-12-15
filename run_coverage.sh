#!/usr/bin/env bash
set -euo pipefail
coverage run --branch --source=myfsm -m pytest
coverage report --fail-under=0