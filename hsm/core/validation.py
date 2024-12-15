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
