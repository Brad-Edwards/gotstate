# File Level Architecture Diagrams

## Module Dependencies

```mermaid
graph TB
    subgraph core[Core Package]
        state[state.py]
        transition[transition.py]
        event[event.py]
        region[region.py]
        machine[machine.py]
    end

    subgraph runtime[Runtime Package]
        executor[executor.py]
        scheduler[scheduler.py]
        monitor[monitor.py]
    end

    subgraph persistence[Persistence Package]
        serializer[serializer.py]
        validator[validator.py]
    end

    subgraph types[Types Package]
        base[base.py]
        type_ext[extensions.py]
    end

    subgraph extensions[Extensions Package]
        hooks[hooks.py]
        sandbox[sandbox.py]
    end

    %% Core dependencies
    state --> region
    transition --> state
    event --> transition
    region --> state
    machine --> state
    machine --> transition
    machine --> event
    machine --> region

    %% Runtime dependencies
    executor --> event
    executor --> transition
    scheduler --> event
    monitor --> machine

    %% Persistence dependencies
    serializer --> machine
    validator --> machine
    validator --> serializer

    %% Types dependencies
    base --> validator
    type_ext --> base

    %% Extensions dependencies
    hooks --> sandbox
    sandbox --> machine
    sandbox --> executor
    sandbox --> serializer

    %% Cross-cutting
    monitor --> |metrics| executor
    monitor --> |metrics| scheduler
    validator --> |validation| hooks
    sandbox --> |security| type_ext
```

## Security Boundaries

```mermaid
graph TB
    subgraph security[Security Boundary]
        subgraph core[Core]
            state[state.py]
            transition[transition.py]
            event[event.py]
            region[region.py]
            machine[machine.py]
        end

        subgraph runtime[Runtime]
            executor[executor.py]
            scheduler[scheduler.py]
            monitor[monitor.py]
        end

        subgraph validation[Validation Layer]
            validator[validator.py]
            validate((Input<br/>Validation))
        end
    end

    subgraph extensions[Extensions]
        hooks[hooks.py]
        sandbox[sandbox.py]
    end

    subgraph persistence[Storage]
        serializer[serializer.py]
        store[(Persistent<br/>Storage)]
    end

    %% Security flow
    extensions --> |sandboxed| sandbox
    sandbox --> |validated| validate
    validate --> |secured| core
    core --> |protected| serializer
    serializer --> |validated| store

    %% Monitoring
    monitor --> |metrics| core
    monitor --> |metrics| runtime
    monitor --> |metrics| extensions

    %% Validation
    validator --> |checks| core
    validator --> |checks| extensions
    validator --> |checks| serializer
```

## Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant Hooks
    participant Sandbox
    participant Validator
    participant Core
    participant Runtime
    participant Storage

    Client->>Hooks: Extension Request
    Hooks->>Sandbox: Isolate Extension
    Sandbox->>Validator: Validate Input
    Validator-->>Sandbox: Input Valid
    Sandbox->>Core: Execute Operation
    Core->>Runtime: Process Event
    Runtime->>Storage: Persist State
    Storage-->>Runtime: Confirm Storage
    Runtime-->>Core: Confirm Processing
    Core-->>Sandbox: Operation Complete
    Sandbox-->>Hooks: Extension Complete
    Hooks-->>Client: Request Complete
```

## Extension Points

```mermaid
graph TB
    subgraph core[Core Extension Points]
        state[State Behavior]
        transition[Transition Logic]
        event[Event Processing]
    end

    subgraph runtime[Runtime Extension Points]
        execution[Execution Strategy]
        scheduling[Timer Management]
        monitoring[Metrics Collection]
    end

    subgraph persistence[Persistence Extension Points]
        format[Storage Format]
        validation[Validation Rules]
        migration[Version Migration]
    end

    subgraph types[Type Extension Points]
        custom[Custom Types]
        conversion[Type Conversion]
        checking[Type Validation]
    end

    hooks[Extension Hooks]
    sandbox[Extension Sandbox]

    %% Extension flow
    hooks --> |manages| sandbox
    sandbox --> |controls| state
    sandbox --> |controls| transition
    sandbox --> |controls| event
    sandbox --> |controls| execution
    sandbox --> |controls| scheduling
    sandbox --> |controls| monitoring
    sandbox --> |controls| format
    sandbox --> |controls| validation
    sandbox --> |controls| migration
    sandbox --> |controls| custom
    sandbox --> |controls| conversion
    sandbox --> |controls| checking

    %% Security
    sandbox --> |isolates| core
    sandbox --> |isolates| runtime
    sandbox --> |isolates| persistence
    sandbox --> |isolates| types
