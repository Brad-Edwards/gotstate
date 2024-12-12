# Hierarchical FSM Library Design Document

The library is modularized into several components, each responsible for specific aspects of the FSM's functionality. The modules are designed with clear responsibilities, minimal dependencies, and well-defined interfaces to facilitate maintainability and scalability.

## Module Overview

| Module      | Primary Responsibilities                                                  | Key Interactions        | Design Considerations                      |
|-------------|---------------------------------------------------------------------------|-------------------------|--------------------------------------------|
| **Core** | FSM lifecycle management, module coordination, event handling logic | Model, Behavior | Immutable after initialization, delegates mutations, clear execution flow |
| **Model**       | State hierarchy definitions, event specifications, transition rules         | Core, Builder           | Immutable post-construction, lock-free reads, efficient traversal |
| **Behavior**    | Guard evaluation, action execution, state entry/exit handling               | Core, Concurrency       | Stateless execution, pure functions, error propagation |
| **Builder** | FSM construction, configuration initialization, initialization sequence | Model, Core | Single-threaded construction, clear error reporting |
| **Concurrency** | Synchronization mechanisms, atomic state management, event delivery and queueing | Core | Lock-free where possible, explicit sync boundaries, atomic operations |
| **Resource**    | Resource lifecycle management, cleanup strategies, memory management         | Concurrency, Core       | Deterministic cleanup, delegated synchronization, resource tracking |
| **Diagnostics** | Error handling and recovery, logging, monitoring, debug information          | All modules             | Lock-free logging, atomic error reporting, structured output |
| **API**         | Public interfaces, type definitions, builder patterns, extension points      | All modules             | Thread-safe interfaces, type safety, comprehensive documentation |
| **Validation** | Cross-cutting validation framework, structural verification, runtime validation, recovery coordination | All modules | Minimal performance impact, non-blocking where possible, clear error reporting |

---

## Module Details

### Core Module

- **Responsibilities**: Acts as the central execution engine for the FSM. Manages the FSM lifecycle and runtime behavior, coordinating interactions between all other modules.
- **Key Features**:
  - Manages execution order and enforces atomicity
  - Defines event handling logic and processing rules
  - Determines event routing through state hierarchy
  - Orchestrates cleanup phases and maintains state consistency

### Model Module

- **Responsibilities**: Defines the structural elements of the FSM, including states, events, and transitions. Manages hierarchical state organization and maintains an immutable structure after initialization.
- **Key Features**:
  - Efficient traversal of the state hierarchy.
  - Lock-free read access to state definitions.
  - Defines timeout events and temporal transition specifications.
  - Validates the structural integrity of the FSM.

### Behavior Module

- **Responsibilities**: Manages the dynamic aspects of the FSM, handling guard condition evaluations, action executions, and state entry/exit behaviors.
- **Key Features**:
  - Implements pure functions for transitions.
  - Ensures atomic evaluation of guards.
  - Executes actions with versioning for consistency.
  - Propagates errors accurately for diagnostics.

### Builder Module

- **Responsibilities**: Provides interfaces for constructing the FSM and initializing sequences.
- **Key Features**:
  - Ensures type safety during FSM construction
  - Performs initialization before FSM instantiation
  - Reports errors with clarity to facilitate debugging
  - Operates in a single-threaded context for initialization

### Concurrency Module

- **Responsibilities**: Implements thread safety mechanisms and provides synchronization primitives.
- **Key Features**:
  - Manages thread-safe event queue operations
  - Provides atomic transition coordination and synchronization
  - Controls event delivery timing and ordering
  - Manages state versions to ensure consistency
  - Enforces lock hierarchies to prevent deadlocks

### Resource Module

- **Responsibilities**: Manages the lifecycle of FSM resources, including memory allocation, deallocation, and cleanup strategies.
- **Key Features**:
  - Ensures deterministic cleanup of resources.
  - Tracks resource usage for efficient management.
  - Manages timer resources and handles cleanup of expired timeouts.
  - Coordinates resource dependencies and cleanup atomicity.

### Diagnostics Module

- **Responsibilities**: Handles error conditions, logging, monitoring, and provides debug information.
- **Key Features**:
  - Implements lock-free logging and atomic error reporting.
  - Integrates with external monitoring tools.
  - Tracks resources and verifies cleanup processes.
  - Reports structured diagnostic information.

### API Module

- **Responsibilities**: Defines public-facing interfaces, type definitions, builder patterns, and extension points for customization.
- **Key Features**:
  - Provides thread-safe interfaces and ensures type safety.
  - Facilitates the creation of custom behaviors through extension points.
  - Includes comprehensive documentation for developers.
  - Supports builder patterns for fluent FSM construction.

---

## Key Design Principles

1. **Separation of Concerns**
   - Maintain clear module boundaries.
   - Minimize cross-module dependencies.
   - Define well-structured interfaces.

2. **Thread Safety**
   - Provide comprehensive concurrency support.
   - Establish a clear threading model.
   - Ensure safe resource sharing and access.

3. **Resource Management**
   - Implement deterministic resource cleanup.
   - Optimize resource utilization.
   - Define a clear ownership and lifecycle model.

4. **Error Handling**
   - Define comprehensive error types.
   - Ensure clear error propagation paths.
   - Provide mechanisms for recovery and rollback.

5. **Extensibility**
   - Support a plugin architecture for custom extensions.
   - Allow customization of behaviors and transitions.
   - Expose well-defined extension points.

---

## Integration Considerations

1. **Module Dependencies**
   - Eliminate circular dependencies.
   - Establish a clear dependency hierarchy.
   - Encourage loose coupling between modules.

2. **Performance**
   - Optimize state transitions for efficiency.
   - Minimize memory allocations during runtime.
   - Utilize lock-free algorithms where appropriate.

3. **Safety**
   - Enforce type safety throughout the library.
   - Ensure memory safety with ownership models.
   - Maintain thread safety in all concurrent operations.

4. **Maintainability**
   - Provide clear and comprehensive documentation.
   - Follow consistent coding patterns and practices.
   - Design for testability with modular components.

---

## Detailed Protocols and Specifications

### Atomic Transition Specifications

#### State Transition Protocol

| Phase | Responsible Modules | Actions and Guarantees |
|--------|-------------------|---------------------|
| Initiation | Concurrency | Acquire event queue lock, Snapshot current state, Capture transition metadata |
| Validation | Behavior | Atomically evaluate guard conditions, Synchronize state data access, Verify preconditions |
| Execution | Concurrency, Behavior | Execute exit actions atomically, Apply state updates atomically, Execute entry actions atomically |
| Completion | Concurrency | Commit state updates, Clean up resources, Release event queue lock |
| Rollback | Concurrency | Restore state from snapshot, Recover resources, Record error state |

#### Transition Atomicity Guarantees

- **State Consistency**
  - Implement version-tagged state updates.
  - Use compare-and-swap (CAS) for state changes.
  - Employ atomic reference counting.

- **Resource Safety**
  - Utilize two-phase resource acquisition.
  - Manage resources capable of rollback.
  - Enforce a deadlock-free resource ordering.

- **Event Ordering**
  - Ensure strict serialization of the event queue.
  - Maintain priority during atomic updates.
  - Guarantee consistent visibility of events.

- **Data Protection**
  - Use copy-on-write for state data access.
  - Update data structures atomically.
  - Apply versioned state storage for concurrency control.

---

### Timeout Event Specifications

#### Timeout Architecture

| Timeout Type | Scope | Recovery Action | Escalation Path |
|-------------|-------|-----------------|-----------------|
| State Entry/Exit | Local State | Forced completion, rollback | Parent state notification |
| Transition | Cross-State | Revert to stable state | Emergency shutdown path |
| Resource Lock | Resource-specific | Force release, compensating action | Resource manager escalation |
| Event Processing | Queue-level | Skip and requeue | Event coordinator notification |

#### Timeout Coordination Protocol

1. **Detection Layer**
   - Heartbeat monitoring
   - Deadline tracking
   - Progress verification
   - Stall detection

2. **Response Layer**
   - Graceful degradation paths
   - Compensation actions
   - State preservation rules
   - Recovery sequences

#### Timeout Hierarchy

| Level | Timeout Policy | Handling Strategy |
|-------|---------------|-------------------|
| Critical | Immediate response | Forced completion |
| Standard | Retry with backoff | Compensating action |
| Background | Best effort | Skip and log |

#### Cross-Module Timeout Integration

| Component | Timeout Responsibility | Coordination Method |
|-----------|----------------------|-------------------|
| Scheduler | Deadline enforcement | Progress tracking |
| Dispatcher | Queue management | Backpressure signals |
| Executor | Action timeouts | Cancellation tokens |
| Resource Manager | Lock timeouts | Forced release protocol |

#### Timeout Management Protocol

| Component | Responsible Module | Actions and Guarantees |
|-----------|-------------------|---------------------|
| Scheduler | Concurrency | Precise timing management, Maintain a priority queue, Support cancellation, Thread-safe scheduling |
| Dispatcher | Core | Deterministic event dispatch, Cleanup of expired timeouts, Preserve event priorities, Integrate with event queue |
| Executor | Behavior | Handle timeouts atomically, Execute state-aware actions, Maintain transition consistency |

#### Timeout Event Properties

- **Precision**
  - Use monotonic clocks for timing.
  - Compensate for clock drift to maintain accuracy.
  - Guarantee minimum latency in event handling.

- **Ordering**
  - Schedule events based on priorities.
  - Ensure FIFO ordering within the same timestamp.
  - Maintain consistent ordering with other events.

- **Lifecycle**
  - Record creation timestamps for timeouts.
  - Handle expiration and provide automatic cleanup.
  - Validate state before executing timeout actions.

---

### Concurrent Cleanup Protocol

#### Cleanup Phase Management

| Phase | Controller Module | Actions and Guarantees |
|--------|------------------|---------------------|
| Preparation | Concurrency | Snapshot resource tracking, Create cleanup barriers, Freeze reference counting |
| Suspension | Core | Suspend event queue processing, Await transition completions, Block new operations |
| Execution | Resource | Release resources in a defined order, Cleanup hierarchical states, Deallocate memory |
| Verification | Concurrency | Verify cleanup completion, Detect resource leaks, Validate reference counts |
| Resumption | Core | Restore event queue, Resume operations, Verify state consistency |

#### Resource Cleanup Ordering

| Level        | Resources Managed                                        | Dependencies   |
|--------------|----------------------------------------------------------|----------------|
| **Primary**     | Active states, event queues, transition data              | None           |
| **Secondary**   | State data, timeout events, history records               | Primary        |
| **Tertiary**    | Monitoring data, debug information, statistical data      | Secondary      |

---

### Thread Safety Patterns

#### Synchronization Hierarchy

| Level        | Scope                    | Implementation                    | Guarantees                          |
|--------------|--------------------------|-----------------------------------|-------------------------------------|
| **Global**      | Entire FSM                | Read-write lock hierarchy           | Deadlock prevention, total ordering |
| **Module**      | Individual components     | Lock-free operations where possible | Inter-module consistency            |
| **Local**       | Specific resources        | Compare-and-swap (CAS) operations   | Resource-level atomicity            |
| **Event**       | Event queues              | Lock-free queue algorithms          | Preservation of event ordering      |

#### Access Patterns

| Resource Type    | Access Pattern             | Protection Mechanism       | Verification Method          |
|------------------|----------------------------|----------------------------|------------------------------|
| **State Data**      | Copy-on-write                  | Atomic references           | Version checks               |
| **Event Queue**     | Multiple producers and consumers | Lock-free algorithms        | Sequence validation          |
| **Configuration**   | Immutable post-initialization  | Read-only enforcement       | Static analysis              |
| **Transitions**     | Two-phase commit               | Optimistic locking          | State version validation     |

#### Deadlock Prevention Protocol

| Level | Strategy | Verification Method |
|-------|----------|-------------------|
| Global | Total Lock Ordering | Static analysis of lock acquisition paths |
| Module | Resource Hierarchy | Lock level validation at runtime |
| Cross-Module | Timeout-based Acquisition | Deadlock detection monitoring |

#### Lock Acquisition Rules

1. **Hierarchical Locking**
   - Strict top-down lock acquisition
   - Parent states before child states
   - Resources ordered by hierarchy level
   - No circular resource dependencies

2. **Resource Classification**
   - Primary: State machine control structures
   - Secondary: State data and event queues
   - Tertiary: Diagnostic and monitoring resources

3. **Timeout Mechanisms**
   - Maximum lock hold times
   - Escalation procedures for timeouts
   - Recovery protocols for failed acquisitions

#### Cross-Module Lock Coordination

| Interaction Type | Lock Order | Conflict Resolution |
|-----------------|------------|-------------------|
| State Transitions | Core → State → Resource | Timeout and rollback |
| Event Processing | Queue → State → Behavior | Skip and requeue |
| Resource Management | Resource → State → Diagnostic | Release and retry |

#### Critical Section Management

- **Global Level**
  - **Components**: Global FSM state, event queue head, transition coordinator.
  - **Mechanism**: Single writer, multiple readers; priority inheritance; ordered locking.
  - **Guarantees**: Deadlock avoidance, consistent state transitions.

- **Module Level**
  - **Components**: State tree nodes, resource registries, schedulers.
  - **Mechanism**: Lock-free operations preferred; atomic updates.
  - **Guarantees**: Efficient concurrency, inter-module consistency.

- **Local Level**
  - **Components**: Individual states, resources, event batches.
  - **Mechanism**: Compare-and-swap operations; atomic reference counting.
  - **Guarantees**: Atomic resource manipulation, prevent race conditions.

- **Event Level**
  - **Components**: Event data, diagnostic info.
  - **Mechanism**: Lock-free queues; atomic counters.
  - **Guarantees**: Accurate event processing, priority maintenance.

#### Thread Safety Guarantees

- **State Transitions**
  - Atomic execution with rollback capabilities.
  - Consistent views of FSM state.
  - Isolation of state changes.

- **Event Processing**
  - Preservation of event order and priorities.
  - Atomic dispatching of events.
  - Lock-free queuing mechanisms.

- **Resource Management**
  - Safe publication and cleanup of resources.
  - Prevention of resource leaks.
  - Clean shutdown procedures.

- **Data Access**
  - Memory visibility across threads.
  - Prevention of race conditions.
  - Enforced ordering of operations.

---

### State Management

#### Complete Rollback Protocol

| Phase | Actions | Guarantees |
|--------|----------|------------|
| Snapshot | Increment version counter, Snapshot state tree, Capture resource references | Atomic capture point |
| Transition | Apply copy-on-write updates, Manage reference counts, Log changes in a journal | Isolated changes |
| Validation | Check state consistency, Verify resource availability, Validate invariants | Atomic verification |
| Commit/Rollback | Atomically swap versions, Exchange references, Replay or discard journal | All-or-nothing outcome |
| Cleanup | Clean up old versions, Release references, Truncate journal | Resource recovery |

| Operation Type | Access Pattern | Synchronization Mechanism |
|----------------|----------------|--------------------------|
| Read | Atomic reference load, Version check, Consistency snapshot | Lock-free |
| Write | Copy-on-write updates, Increment version, Swap references | Atomic operations |
| Transition | Double-buffered states, Version barriers, Exchange references | Transactional mechanisms |

---

### State Hierarchy Model

#### Structure and Relationships

| Relationship Type | Properties | Constraints |
|------------------|------------|-------------|
| Parent-Child | One parent, multiple children | No circular references, max depth limit configurable |
| Sibling | Share same parent | Concurrent transitions allowed |
| Root | No parent, top of hierarchy | Single root per FSM instance |
| Composite | Contains substates | Must define entry/exit policies |
| Atomic | No substates | Terminal state in hierarchy |

#### State Activation Rules

1. **Entry Sequence**
   - Parent states must be active before children
   - Default child state activation on parent entry
   - Entry action execution order: parent-to-child
   - Atomic activation guarantee across hierarchy levels

2. **Exit Sequence**
   - Child states must exit before parents
   - Exit action execution order: child-to-parent
   - Resource cleanup ordered by hierarchy level
   - Complete subtree deactivation guarantee

---

### Hierarchical Event Processing

#### Event Propagation Model

| Direction | Behavior | Use Case |
|-----------|----------|----------|
| Bottom-Up | Child to parent propagation | Event bubbling for unhandled events |
| Top-Down | Parent to children propagation | Broadcast events, global state changes |
| Horizontal | Between sibling states | Peer communication, coordination |

#### Event Handling Priority

1. **Processing Order**
   - Local state handlers first
   - Child states before parent states
   - Sibling states in definition order
   - Parent states as fallback

2. **Concurrent Event Processing**
   - Independent subtrees process concurrently
   - Sibling state events may be parallel
   - Parent-child event ordering preserved
   - Cross-hierarchy synchronization points

---

### Hierarchical Transition Management

#### Transition Types

| Type | Scope | Synchronization Requirements |
|------|-------|----------------------------|
| Local | Within single state | Standard transition protocol |
| Vertical | Across hierarchy levels | Parent-child coordination |
| Horizontal | Between siblings | Sibling state coordination |
| Cross-Tree | Different branches | Full hierarchy coordination |

#### Transition Coordination

1. **Scope Determination**
   - Identify lowest common ancestor
   - Calculate affected subtrees
   - Determine required synchronization
   - Plan resource management

2. **Execution Protocol**

   ```plaintext
   For each transition:
   1. Lock affected subtree scope
   2. Exit descendant states bottom-up
   3. Execute transition actions
   4. Enter new states top-down
   5. Release locks in reverse order
   ```

---

### Hierarchical Resource Management

#### Resource Scoping

| Scope Level | Ownership | Cleanup Timing |
|------------|-----------|----------------|
| Local | Single state | State exit |
| Shared | State subtree | Common ancestor exit |
| Global | Entire hierarchy | FSM shutdown |

#### Resource Inheritance

1. **Access Patterns**
   - Child states can access parent resources
   - Resource visibility follows hierarchy
   - Override mechanisms for child states
   - Version tracking across hierarchy

2. **Cleanup Protocol**

   ```plaintext
   During state/subtree exit:
   1. Release child-specific resources
   2. Remove overrides of parent resources
   3. Decrement shared resource references
   4. Clean up unreferenced shared resources

---

### Cross-Hierarchy Operations

#### Synchronization Points

1. **Mandatory Sync Points**
   - Cross-tree transitions
   - Global event processing
   - Resource cleanup initiation
   - Error recovery coordination

2. **Optional Sync Points**
   - Sibling state coordination
   - Resource optimization
   - Diagnostic snapshots
   - Performance monitoring

#### Operation Guarantees

1. **Consistency Guarantees**
   - Hierarchical state consistency
   - Event processing ordering
   - Resource lifecycle management
   - Error propagation paths

2. **Performance Considerations**
   - Parallel execution boundaries
   - Lock granularity optimization
   - Resource sharing efficiency
   - Event processing throughput

---

### Synchronization Architecture

#### Lock Hierarchy Specification

| Level                | Components                                      | Priority |
|----------------------|-------------------------------------------------|----------|
| **Level 0 (Highest)**   | FSM version control, event queue head, transition coordinator | Maximum  |
| **Level 1**             | State tree nodes, resource registries, schedulers       | High     |
| **Level 2**             | Individual states, resources, event batches             | Medium   |
| **Level 3 (Lowest)**    | State data, event payloads, diagnostics                 | Low      |

#### Memory Barrier Requirements

| Barrier Type  | Operations Affected                         | Purpose                        |
|---------------|---------------------------------------------|--------------------------------|
| **Store**        | State updates, queue operations, reference changes | Ensure visibility to other threads |
| **Load**         | State reads, event processing, resource access      | Prevent reordering of operations |
| **Full**         | Transition completions, cleanup initiation, version changes | Provide complete isolation |

#### Priority Inheritance Protocol

- **High-Priority Wait Scenarios**
  - Temporarily boost priority of lock holders.
  - Implement priority inheritance mechanisms.
  - Restore original priorities after lock release.

- **Nested Locks**
  - Track lock acquisition chains.
  - Cascade priorities to prevent deadlocks.
  - Resolve lock chains based on predefined ordering.

- **Resource Access**
  - Apply priority ceilings to shared resources.
  - Boost priorities when accessing critical resources.
  - Restore priorities upon completion of operations.

---

### Resource Management

#### Resource Pool Architecture

| Pool Type | Scope | Management Strategy | Recovery Policy |
|-----------|-------|-------------------|-----------------|
| State Resources | Per State Hierarchy | Hierarchical allocation with inheritance | Cascade cleanup |
| Event Handlers | Global FSM | Pre-allocated handler pools | Dynamic scaling |
| Transition Workers | Global FSM | Fixed-size thread pool | Work stealing |

#### Pool Management Protocols

1. **Allocation Strategies**
   - Static pools for critical resources
   - Dynamic pools for variable workloads
   - Hybrid pools for state-specific resources

2. **Resource Sharing Rules**
   - Cross-hierarchy resource sharing policies
   - Pool access priority levels
   - Resource borrowing constraints

#### Pool Integration Points

| Module | Pool Integration | Scaling Policy |
|--------|-----------------|----------------|
| Core | Worker thread management | Fixed size |
| State | State-specific resource pools | Hierarchical |
| Event | Event handler pools | Dynamic |

#### Timeout Cancellation Protocol

| State | Actions | Cleanup Procedure |
|--------|----------|------------------|
| Pending | Remove from scheduler queue, Cancel timers, Release associated resources | Immediate cleanup |
| Active | Mark as cancelled, Prevent triggering, Queue for cleanup | Deferred cleanup |
| Triggered | Complete handling, Prevent cascading effects, Perform standard cleanup | Standard cleanup |

---

## Validation Architecture

### Component Overview

| Component | Purpose | Integration Points |
|-----------|---------|-------------------|
| Validator | Coordinates validation operations, manages validation rules | Builder, Core, Model |
| Rule Engine | Executes validation rules, aggregates results | Model, Behavior |
| Recovery Coordinator | Manages recovery actions, coordinates with error handling | Core, Resource |
| Context Manager | Maintains validation context, provides access to state information | All modules |

### Validation Integration Strategy

1. **Construction Phase Integration**
   - Extends Builder module capabilities
   - Integrates with Model validation
   - Enhances configuration verification

2. **Runtime Integration**
   - Leverages existing transition protocols
   - Extends event processing pipeline
   - Enhances resource management checks

3. **Recovery Integration**
   - Augments existing error handling
   - Extends rollback capabilities
   - Enhances diagnostic reporting

### Hierarchical Validation Rules

#### Structure Validation

| Rule Type | Validation Criteria | Recovery Action |
|-----------|-------------------|-----------------|
| Depth Limits | Maximum hierarchy depth, branching factors | Reject configuration |
| State Relations | Parent-child validity, no cycles, unique paths | Configuration error |
| Resource Patterns | Inheritance chains, scope definitions | Resource reallocation |
| Event Handlers | Coverage completeness, propagation paths | Handler adjustment |

#### Runtime Validation Rules

| Aspect | Validation Rules | Recovery Strategy |
|--------|-----------------|-------------------|
| State Activation | Parent-before-child, completion guarantees | Revert to last valid state |
| Transition Flow | Cross-hierarchy consistency, lock ordering | Rollback transaction |
| Resource Access | Inheritance patterns, cleanup ordering | Resource reset |
| Event Processing | Order preservation, propagation correctness | Event requeue |

---

## Cross-Module Operation Protocols

### State Transition Coordination

- **Core Module Initiates**:
  - Requests transition based on events.
  - Coordinates with Concurrency and Behavior modules.

- **Concurrency Module Manages**:
  - Handles synchronization and atomic operations.
  - Ensures thread safety during transitions.

- **Behavior Module Executes**:
  - Evaluates guards and performs actions.
  - Executes entry and exit behaviors.

- **Resource Module Updates**:
  - Manages resources affected by the transition.
  - Ensures proper acquisition and release.

### Event Processing Flow

1. **Event Generation**:
   - Events are generated internally or received from external sources.
   - Enqueued in a thread-safe, lock-free event queue.

2. **Event Dispatching**:
   - Core module dispatches events to the appropriate handlers.
   - Maintains event ordering and priority.

3. **Event Handling**:
   - Behavior module processes the event.
   - Executes associated actions and triggers state transitions if necessary.

4. **Concurrency Control**:
   - Concurrency module ensures thread-safe processing.
   - Manages synchronization and atomicity during event handling.

### Validation Coordination

- **Validator Module Coordinates**:
  - Manages validation rule execution
  - Coordinates with Core for transition validation
  - Interfaces with Resource for cleanup validation

- **Core Module Integrates**:
  - Provides validation hooks in transition pipeline
  - Coordinates recovery actions
  - Manages validation context

- **Model Module Supports**:
  - Provides structural validation information
  - Maintains validation metadata
  - Supports rule evaluation

- **Resource Module Extends**:
  - Includes validation in cleanup protocols
  - Provides resource state validation
  - Supports recovery operations

- **Builder Module Enforces**:
  - Exposes FSM structure to Validation module during construction
  - Provides construction-time hooks for validation checks
  - Propagates validation results during build process
  - Fails construction if validation requirements not met
