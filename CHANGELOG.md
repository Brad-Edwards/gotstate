# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-02-24

### Added

- Complete redesign of the library architecture
- New core package with improved state machine implementation
- Hierarchical state organization
- Support for entry and exit actions
- Guard conditions for transitions
- Composite states and regions
- Strong typing throughout the codebase
- AsyncStateMachine extension for asynchronous state machines
- Comprehensive unit tests

### Changed

- Package structure completely reorganized
- API simplified and made more intuitive
- Improved error handling and debugging
- Switched to Poetry for dependency management
- Updated GitHub workflows for testing, linting, and security
- Documentation improved with better examples

### Removed

- Legacy code from version 1.x
- Deprecated features and functions

## [1.2.3] - 2023-01-25

### Fixed

- Fixed a bug where transitions were not correctly processing guard conditions
- Updated dependencies to latest versions

## [1.2.2] - 2022-12-10

### Added

- Support for Python 3.11
- Improved type hints

### Fixed

- Fixed a race condition in the event queue processing

## [1.2.1] - 2022-10-15

### Fixed

- Fixed an issue with nested state handling
- Documentation improvements

## [1.2.0] - 2022-08-20

### Added

- Added support for history states
- Added new examples

### Changed

- Improved error messages
- Better documentation

## [1.1.0] - 2022-05-15

### Added

- Added basic support for nested states
- Added state callbacks

### Changed

- Improved event processing

## [1.0.0] - 2022-03-01

### Added

- Initial release of gotstate
- Basic state machine functionality
- Event-driven transitions
- Simple state and transition management

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
