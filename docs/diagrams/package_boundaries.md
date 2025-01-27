# Package Boundaries and External Interactions

```mermaid
graph TB
    subgraph gotstate[gotstate Package]
        core[HFSM Core]
    end

    client[Client Code]
    api[Public API]
    storage[(Storage)]
    os[OS Services]
    types[Python Type System]
    logging[Logging System]

    client --> |uses| api
    api --> |exposes| core
    core --> |persistence| storage
    core --> |concurrency| os
    core --> |type checking| types
    core --> |diagnostics| logging

    %% External dependency styling
    linkStyle 2 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
    linkStyle 3 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
    linkStyle 4 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
    linkStyle 5 stroke:#666,stroke-width:2,stroke-dasharray: 5 5;
