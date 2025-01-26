# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-22

### Added
- Hierarchical state machine implementation
- Core functionality
  - State management and transitions
  - Event handling
  - Guard conditions
  - Transition actions
  - State data lifecycle
- Validation system
- Plugin architecture
- Test suite
- Type hints
- Error handling
- Documentation and examples
- CI/CD pipeline
- Development tools

### Changed
- Migrated from Rust to Python implementation
- Stabilized APIs
- Finalized plugin interface

### Security
- Security scanning in CI
- Input validation
- Runtime checks

## [1.0.1] - 2024-12-22
### Added
- Moved package to `gotstate` and updated `pyproject.toml`

## [1.0.2] - 2024-12-22
### Changed
- Updated `pyproject.toml`
- Updated `sonar-project.properties`

## [1.0.3] - 2024-12-22
### Changed
- Updated project URLs in package metadata

## [1.0.4] - 2025-01-26
### Fixed
- Updated package configuration to use gotstate instead of hsm
