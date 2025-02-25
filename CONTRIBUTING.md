# Contributing to GotState

Thank you for your interest in contributing to GotState! This document provides guidelines and instructions for contributing to this project.

## Setting Up Development Environment

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/yourusername/gotstate.git
   cd gotstate
   ```

2. Set up a virtual environment using Poetry:

   ```bash
   # Install Poetry if you haven't already
   # curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install --with dev
   
   # Activate the virtual environment
   poetry shell
   ```

3. Install pre-commit hooks:

   ```bash
   poetry run pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests for them.

3. Run the tests to ensure everything works:

   ```bash
   poetry run pytest
   ```

4. Run linting checks:

   ```bash
   poetry run black gotstate tests
   poetry run isort gotstate tests
   poetry run flake8 gotstate tests
   poetry run mypy gotstate
   ```

5. Commit your changes with a descriptive message:

   ```bash
   git commit -m "Add feature X"
   ```

6. Push your branch to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request from your fork to the main repository.

## Pull Request Guidelines

- Follow the Python code style conventions (PEP 8).
- Include unit tests for new features or bug fixes.
- Update documentation if necessary.
- Make sure all tests pass before submitting a pull request.
- Keep pull requests focused on a single feature or bug fix.

## Code Style

We use the following tools to maintain code quality:

- `black` for code formatting
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

## Testing

Please write tests for any new features or bug fixes. We use `pytest` for testing.

To run tests:

```bash
poetry run pytest
```

To run tests with coverage:

```bash
poetry run pytest --cov=gotstate
```

## Documentation

If your changes affect the API or add new features, please update the documentation accordingly.

## License

By contributing to GotState, you agree that your contributions will be licensed under the project's MIT License.
