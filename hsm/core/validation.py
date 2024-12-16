# hsm/core/validation.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
from typing import List

from hsm.interfaces.abc import AbstractValidator
from hsm.interfaces.types import ValidationResult


class Validator(AbstractValidator):
    def validate_structure(self) -> List[ValidationResult]:
        raise NotImplementedError

    def validate_behavior(self) -> List[ValidationResult]:
        raise NotImplementedError

    def validate_data(self) -> List[ValidationResult]:
        raise NotImplementedError
