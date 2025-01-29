# Package Boundaries and External Interactions

```mermaid
graph TB
    subgraph gotstate[gotstate Package]
        subgraph security[Security Boundary]
            validate((Input<br/>Validation))
            core[HFSM Core]
            protect((State<br/>Protection))
        end
        
        subgraph config[Configuration Boundary]
            config_validate((Config<br/>Validation))
            config_store((Config<br/>Storage))
            config_apply((Config<br/>Application))
        end
    end

    client[Client Code]
    api[Public API]
    storage[(Storage)]
    os[OS Services]
    types[Python Type System]
    logging[Logging System]
    env[Environment]
    config_file[Config Files]

    %% Control flow
    client --> |uses| api
    api --> |validates| validate
    validate --> |secured| core
    core --> |protected| protect
    
    %% Configuration flow
    client --> |configures| config_validate
    env --> |provides| config_validate
    config_file --> |provides| config_validate
    config_validate --> |validates| config_store
    config_store --> |applies| config_apply
    config_apply --> |configures| core
    
    %% External interactions
    protect --> |validated persistence| storage
    protect --> |managed threads| os
    validate --> |type checking| types
    protect --> |filtered logs| logging

    %% External dependency styling
    linkStyle 4 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
    linkStyle 5 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
    linkStyle 6 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
    linkStyle 7 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
```

## Configuration Boundaries

The configuration system maintains strict boundaries for security and reliability:

1. Configuration Sources

- Environment variables are validated before use
- Configuration files are validated against schema
- Programmatic configuration is type-checked
- Runtime changes are validated atomically

2. Configuration Storage

- Configurations are immutable once validated
- Changes create new configuration instances
- History of changes is maintained
- Rollback points are preserved

3. Configuration Application

- Changes are applied atomically
- Components are notified of changes
- State is preserved during updates
- Failed changes are rolled back

4. Security Controls

- Configuration values are validated
- Sensitive data is protected
- Access is controlled
- Changes are audited
