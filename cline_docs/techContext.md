# Technical Context

## Technologies Used

### Core Technologies

- Python ≥3.8
- Type hints and runtime type checking
- Threading for parallel execution
- JSON/YAML for serialization

### Development Tools

- Pre-commit hooks (.pre-commit-config.yaml)
- Pytest for testing (pytest.ini)
- Bandit for security scanning (bandit.yaml)
- Sonar for code quality (sonar-project.properties)

### Project Structure

```
gotstate/
├── core/           # Core HFSM implementation
├── runtime/        # Execution engine
├── persistence/    # State persistence
├── types/         # Type system
└── extensions/    # Extension framework
```

## Development Setup

### Dependencies

- requirements.txt: Core dependencies
- requirements-dev.txt: Development dependencies
- pyproject.toml: Project metadata and build config

### Testing Infrastructure

- Unit tests in tests/unit/
- Integration tests in tests/integration/
- Coverage reports in coverage-reports/

### Documentation

- Architecture docs in docs/architecture/
- Requirements in docs/hfsm_requirements/
- Reference materials in reference/

## Technical Constraints

### UML Compliance

- Strict adherence to UML state machine specification
- Compliant state transitions
- Standard pseudostate implementations

### Performance Requirements

- Predictable scaling with state complexity
- Efficient event processing
- Resource-bounded execution
- Memory management controls

### Security Boundaries

- Input validation at API boundaries
- State isolation enforcement
- Resource limit controls
- Extension sandboxing

### Type System

- Static type checking
- Runtime type validation
- Extension type safety
- Type conversion interfaces

## Extension Points

### Core Extensions

- State behavior customization
- Event processing hooks
- Persistence strategies
- Monitoring interfaces

### Integration Points

- Storage system interfaces
- Operating system integration
- Logging system hooks
- Type system extensions

## Configuration

### Package Configuration

```python
{
    'validation': {
        'strict_mode': True,
        'runtime_checks': True
    },
    'security': {
        'state_isolation': True,
        'input_validation': True
    },
    'resources': {
        'max_memory': '1GB',
        'max_threads': 4,
        'max_states': 1000
    }
}
```

### Environment Variables

- GOTSTATE_STORAGE_BACKEND
- GOTSTATE_STORAGE_PATH
- GOTSTATE_RUNTIME_EXECUTOR

### File-based Config

- YAML/JSON configuration files
- Environment-specific settings
- Override capabilities
