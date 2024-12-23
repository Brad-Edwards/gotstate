# gotstate

[![Security](https://github.com/KeplerOps/gotstate/actions/workflows/security.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/security.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Brad-Edwards_gotstate&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Brad-Edwards_gotstate)
[![Tests](https://github.com/KeplerOps/gotstate/actions/workflows/test.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/test.yml)
[![Lint](https://github.com/KeplerOps/gotstate/actions/workflows/lint.yml/badge.svg)](https://github.com/KeplerOps/gotstate/actions/workflows/lint.yml)

A hierarchical finite state machine (HFSM) library for Python, focusing on reliability and ease of use.

## Features

- Hierarchical state machines with composite states
- Type-safe state and event handling
- Thread-safe event processing
- Guard conditions and transition actions
- State data management with lifecycle hooks
- Timeout events
- History states (both shallow and deep)
- Error handling
- Activation hooks for monitoring
- Plugin system

## Status

**Version 1.0.2**

Initial release!

## Design Philosophy

`gotstate` is designed with the following principles:

- **Safety**: Runtime validation and type checking
- **Clarity**: Intuitive API design
- **Reliability**: Built for real-world applications
- **Performance**: Minimal overhead
- **Flexibility**: Extensible through plugins

## Installation

Install using pip:

```bash
pip install gotstate
```

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for full dependencies

## Documentation

Documentation is available in the `docs/` directory:
- API Reference
- Usage Guide
- Examples

## License

Licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines in CONTRIBUTING.md.

## Security

This package follows Python security best practices.
