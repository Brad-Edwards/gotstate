# hsm/tests/conftest.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import pytest


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line("markers", "stress: mark test as a stress test")
    config.addinivalue_line("markers", "property: mark test as a property-based test")
    config.addinivalue_line("markers", "memory: mark test as a memory test")
