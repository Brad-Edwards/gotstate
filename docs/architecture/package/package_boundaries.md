# Package Boundaries and External Interactions

```mermaid
graph TB
    subgraph gotstate[gotstate Package]
        subgraph security[Security Boundary]
            validate((Input<br/>Validation))
            core[HFSM Core]
            protect((State<br/>Protection))
        end
    end

    client[Client Code]
    api[Public API]
    storage[(Storage)]
    os[OS Services]
    types[Python Type System]
    logging[Logging System]

    %% Control flow
    client --> |uses| api
    api --> |validates| validate
    validate --> |secured| core
    core --> |protected| protect
    
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
