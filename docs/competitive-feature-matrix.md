# Comprehensive State Machine / Statechart Library Feature Matrix

A competitive analysis of gotstate against the leading state machine and statechart libraries across different ecosystems.

## Libraries Compared

| Library | Language | Ecosystem |
|---------|----------|-----------|
| **gotstate** | Python | Standalone |
| **XState v5** | JavaScript/TypeScript | Node.js, React, Vue, Svelte |
| **Boost.SML** | C++14 | Header-only, Boost ecosystem |
| **Boost.MSM** | C++ | Boost ecosystem |
| **Spring Statemachine** | Java | Spring Framework |
| **SCXML** | XML | W3C Standard (implementation-agnostic) |

---

## 1. State Types

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Atomic states** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Composite / hierarchical** | Yes | Yes | Yes (sub-machines) | Yes (submachines) | Yes | Yes |
| **Parallel / orthogonal regions** | Yes | Yes (`type: 'parallel'`) | Yes (multiple initial `*`) | Yes (orthogonal regions) | Yes (regions, fork/join) | Yes (`<parallel>`) |
| **Initial state** | Yes | Yes (`initial`) | Yes (`*` prefix) | Yes (`initial_state`) | Yes | Yes (`<initial>`) |
| **Final state** | Yes | Yes (`type: 'final'`) | Yes (`X` terminate) | Yes (terminate pseudo-state) | Yes | Yes (`<final>`) |
| **History - shallow** | Yes | Yes | Yes (`(H)`) | Yes | Yes (`History.SHALLOW`) | Yes |
| **History - deep** | Yes | Yes | No | Yes (`H*`) | Yes (`History.DEEP`) | Yes |
| **Choice pseudo-state** | No | Yes (guarded arrays) | Yes (via guards) | Yes (implicit via guard ordering) | Yes | No (use guards) |
| **Junction pseudo-state** | No | No | No | No explicit | Yes | No |
| **Fork pseudo-state** | No | No (implicit via parallel) | No (implicit) | Yes (explicit fork entry) | Yes | No (implicit) |
| **Join pseudo-state** | No | No (implicit via parallel) | No | Yes | Yes | No (implicit) |
| **Entry point pseudo-state** | No | No | No | Yes | Yes | No |
| **Exit point pseudo-state** | No | No | No | Partial | Yes | No |

## 2. Transition Types

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **External transitions** | Yes | Yes (default) | Yes (default) | Yes | Yes | Yes (default) |
| **Internal transitions** | Yes | Yes (`internal: true`) | Yes | Yes | Yes | Yes (`type="internal"`) |
| **Local transitions** | No | Yes | No | No | No | Yes (`type="local"`) |
| **Self-transitions** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Eventless / automatic** | No | Yes (`always`) | Yes (anonymous events) | Yes (completion transitions) | Yes (triggerless) | Yes (no `event` attr) |
| **Delayed / timed** | No | Yes (`after: { delay }`) | No | No | Yes (timer-based) | Yes (`<send delay>`) |
| **Wildcard transitions** | No | Yes (`*` event) | Yes (`_` catch-all) | No | No | Yes (`*` descriptor) |
| **Guarded transitions** | Yes | Yes (`guard`/`cond`) | Yes (`[ guard ]`) | Yes | Yes | Yes (`cond` attr) |
| **Compound / inter-level** | Partial | Yes | Limited | Yes | Yes | Yes |
| **Re-enter transitions** | No | Yes (`reenter: true`) | Yes | Yes | No | No |

## 3. Actions

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Entry actions** | Yes | Yes (`entry`) | Yes (on_entry) | Yes | Yes | Yes (`<onentry>`) |
| **Exit actions** | Yes | Yes (`exit`) | Yes (on_exit) | Yes | Yes | Yes (`<onexit>`) |
| **Transition actions** | Yes | Yes (`actions`) | Yes (`/ action`) | Yes | Yes | Yes (in `<transition>`) |
| **Activity / do actions** | No | No (use `invoke`) | No | No | No | No |
| **Raise event** | No | Yes (`raise()`) | Yes (`process(event)`) | Yes | Yes | Yes (`<raise>`) |
| **Assign / context update** | No | Yes (`assign()`) | No (DI) | Yes (state data) | Yes | Yes (`<assign>`) |
| **Send (external)** | No | Yes (`sendTo()`) | No | No | Yes | Yes (`<send>`) |
| **Cancel delayed** | No | Yes (`cancel()`) | No | No | No | Yes (`<cancel>`) |
| **Log** | Yes (logging) | Yes (custom) | Yes (logging policy) | No | Yes (listeners) | Yes (`<log>`) |
| **Conditional execution** | Yes (guards) | Yes | Yes | Yes | Yes | Yes (`<if>`) |
| **Foreach / iteration** | No | No | No | No | No | Yes (`<foreach>`) |
| **Script execution** | No | No | No | No | No | Yes (`<script>`) |

## 4. Guards / Conditions

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Boolean guards** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Composable: AND** | No | Yes (`and([...])`) | Manual | Manual | Manual | Manual |
| **Composable: OR** | No | Yes (`or([...])`) | Manual | Manual | Manual | Manual |
| **Composable: NOT** | No | Yes (`not(guard)`) | Manual | Manual | Manual | Manual |
| **In-state guard** | No | Yes (`stateIn()`) | Yes (`is(state)`) | No | No | Yes (`In()`) |
| **SpEL / expression guards** | No | No | No | No | Yes (SpEL) | Yes (ECMAScript) |

## 5. Event Handling

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Event queuing** | Yes | Yes (actor mailbox) | No (synchronous) | No (synchronous) | Yes | Yes (dual queues) |
| **Internal events** | Yes | Yes (raise) | Yes (process) | Yes | Yes | Yes (internal queue) |
| **External events** | Yes | Yes (send) | Yes | Yes | Yes | Yes (external queue) |
| **Deferred events** | Yes | No | Yes (defer/process) | Yes | Yes | Partial (implicit) |
| **Event broadcasting** | No | Yes (actor system) | No | No | Yes (listeners) | Yes (`<send>`) |
| **Priority: internal before external** | No | Yes | N/A (synchronous) | N/A | N/A | Yes |
| **Wildcard event matching** | No | Yes (`*`) | Yes (`_`) | No | No | Yes (`*`) |
| **Event payloads / data** | Yes | Yes (event objects) | Yes | Yes | Yes (headers + payload) | Yes (`<param>`) |
| **Error events** | Partial | Yes (onError) | Yes (exception events) | No | Yes (interceptor) | Yes (`error.*`) |

## 6. Hierarchical Features

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Nested states** | Yes (arbitrary depth) | Yes (arbitrary depth) | Yes (composite) | Yes (submachines) | Yes | Yes |
| **Inter-level transitions** | Partial | Yes | Limited | Yes | Yes | Yes |
| **Transition conflict policy** | Not implemented | SCXML-compliant (child priority) | Last-match wins | Last-in-table first | Configurable | Child state priority |

## 7. Communication and Actor Model

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Actor model** | No | Yes (first-class) | No | No | No | Partial (`<invoke>`) |
| **Invoke services** | No | Yes (`invoke`) | No | No | No | Yes (`<invoke>`) |
| **Spawn actors** | No | Yes (`spawn`) | No | No | No | No |
| **Parent-child communication** | No | Yes (`sendParent`, `sendTo`) | No | No | No | Yes (`#_parent`) |
| **Inter-machine messaging** | No | Yes (actor refs) | No | No | Yes (distributed) | Yes (`<send target>`) |
| **Done event propagation** | No | Yes (`onDone`) | No | No | Yes (listeners) | Yes (`done.invoke.id`) |

## 8. Persistence and Serialization

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **State snapshots** | No | Yes (`getPersistedSnapshot()`) | No | No | Yes (Persister) | No (impl-specific) |
| **Restore from snapshot** | No | Yes (`createActor(machine, { snapshot })`) | No | No | Yes | No (impl-specific) |
| **JSON serialization** | No | Yes (native) | No | No | Yes (JPA, Redis) | XML (native) |
| **Database persistence** | No | Via user code | No | No | Yes (JPA, Redis, MongoDB) | No (impl-specific) |
| **Deep persistence (child actors)** | No | Yes (recursive) | No | No | Yes | No |

## 9. Visualization and Inspection

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Visual editor** | No | Yes (Stately Studio) | No | No | Yes (Eclipse Papyrus) | Various (Qt SCXML) |
| **Runtime inspector** | No | Yes (`@statelyai/inspect`) | No | No | Yes (Actuator metrics) | Impl-specific |
| **State diagram generation** | No | Yes (auto from code) | No (manual PlantUML) | No | Yes (UML Papyrus) | Inherent (XML) |
| **Logging** | Yes | Yes (inspect API) | Yes (logging policy) | No | Yes (listeners) | Yes (`<log>`) |

## 10. Testing

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Testing utilities** | Yes (pytest suite) | Yes (`xstate/graph`) | Yes (`sml::testing::sm`) | No | Yes (TestPlanBuilder) | W3C test suite (IRP) |
| **Model-based test generation** | No | Yes (auto path gen) | No | No | No | No |
| **Set arbitrary state** | No | Yes (via snapshot) | Yes (`set_current_states`) | No | Yes (via persistence) | N/A |
| **State inspection** | Yes | Yes (`state.matches()`, `state.can()`) | Yes (`is()`, `visit_current_states`) | Yes (is_flag_active) | Yes (query methods) | Impl-specific |

## 11. Extended State / Context Data

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Context / extended state** | Yes (state data) | Yes (`context`) | No (DI) | Yes (state data) | Yes (extended state vars) | Yes (`<datamodel>`) |
| **Typed context** | Yes (Python types) | Yes (TypeScript generics) | Yes (compile-time) | Yes (compile-time) | Yes (Java generics) | Partial |
| **Context update actions** | Partial | Yes (`assign()`) | Via DI mutation | Direct mutation | Via extended state API | Yes (`<assign>`) |
| **Input data** | Yes | Yes (`input`) | Via constructor DI | Via constructor | Via Spring beans | Yes (`<data>`, `<param>`) |

## 12. SCXML Compliance

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **SCXML compliance** | None | High (algorithm-based) | None (UML 2.0) | None (UML 2.0) | None (UML-inspired) | Definitive |
| **SCXML import/export** | No | v4 partial; v5 dropped | No | No | No | Native format |
| **W3C algorithm** | No | Yes | No | No | No | Yes (normative) |

## 13. Error Handling

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Error events** | Partial | Yes (`onError`) | Yes (exception events) | No | Yes (interceptor) | Yes (`error.*`) |
| **Guard exception handling** | Custom | Transition not taken | SM stays in current state | Unspecified | Interceptor-based | Error event raised |
| **Action exception handling** | Custom | Via `onError` | Move to new state | Unspecified | Error actions | Error event raised |

## 14. Unique and Differentiating Features

| Feature | gotstate | XState (JS/TS) | Boost.SML (C++14) | Boost.MSM (C++) | Spring SM (Java) | SCXML (W3C) |
|---|---|---|---|---|---|---|
| **Design-by-Contract** | Yes (unique) | No | No | No | No | No |
| **Dynamic modification** | Yes (MachineModifier) | No (immutable machines) | No | No | No | No |
| **Extension sandbox** | Yes (resource-limited) | No | No | No | No | No |
| **Thread safety (reentrant lock)** | Yes (RLock) | N/A (single-threaded) | Yes (configurable policy) | No | Yes (Spring-managed) | Impl-specific |
| **Deferred events** | Yes | No | Yes | Yes | Yes | Partial |
| **Actor model** | No | Yes (first-class) | No | No | No | Partial |
| **Distributed SM** | No | No | No | No | Yes (Zookeeper) | No |
| **Visual editor / studio** | No | Yes (best-in-class) | No | No | Yes (Papyrus) | Various |
| **Framework integrations** | No | React, Vue, Svelte, Solid | None | Boost | Spring IoC, Security | Impl-specific |
| **Reactive support** | No | Yes (observable) | No | No | Yes (3.0+) | No |
| **Zero dependencies** | Yes | Yes | Yes | Boost headers | Spring Framework | N/A |
| **Tags / metadata** | No | Yes (`tags`, `meta`) | No | No | No | No |
| **Routable states** | No | Yes (v5.28+) | No | No | No | No |

---

## gotstate Competitive Position

### Where gotstate leads

- **Design-by-Contract validation** — no other state machine library integrates contract-based invariant checking. This is a genuinely novel capability for safety-critical and correctness-focused applications.
- **Dynamic runtime modification** — `MachineModifier` enables atomic structural changes to running machines. XState machines are immutable by design; other libraries either don't support this or require full reconstruction.
- **Extension sandbox** — resource-limited plugin execution with memory and CPU constraints is unique across all compared libraries.
- **Thread safety with RLock** — proper reentrant lock-based concurrent access (JavaScript is single-threaded, Boost.SML requires policy configuration, Boost.MSM has no built-in thread safety).
- **Deferred events** — gotstate implements deferred events, which XState notably lacks. This is a meaningful UML/SCXML feature for complex event-driven systems.

### Where gotstate trails

- **Eventless / automatic transitions** — all major competitors support completion or `always` transitions; gotstate does not.
- **Actor model / invoke** — XState's actor model is a major architectural differentiator that gotstate lacks entirely.
- **Persistence and serialization** — XState has native snapshot support; Spring SM integrates with JPA, Redis, MongoDB. gotstate has no persistence layer.
- **Visualization and tooling** — XState's Stately Studio is best-in-class. gotstate has no visual editor, runtime inspector, or diagram generation.
- **SCXML compliance** — XState closely follows the W3C algorithm. gotstate has no SCXML alignment.
- **Composable guards** — XState provides `and()`, `or()`, `not()` guard combinators. gotstate guards are simple boolean functions.
- **Pseudostates** — Spring SM supports the full UML pseudostate complement (choice, junction, fork, join, entry/exit points). gotstate supports none of these.
- **Delayed / timed transitions** — XState and SCXML have first-class timer support. gotstate has no built-in timing.
- **Wildcard event matching** — XState and Boost.SML support catch-all event patterns. gotstate does not.
- **Model-based testing** — XState can automatically generate test paths from machine definitions. gotstate has a pytest suite but no generative testing.

### Estimated operational capability coverage

gotstate implements approximately **36%** of the feature surface covered by the union of all compared libraries. The architecture is clean and the implemented primitives (states, transitions, guards, actions, hierarchy, parallel regions, history, deferred events) are well-structured. The primary gap is the **integration layer** — the code that makes hierarchical states, parallel regions, history, timers, and pseudostates work together through the machine's event processing loop. This integration layer is what separates "a library with state machine classes" from "a working statechart engine."

### Strategic recommendations

1. **Highest impact additions**: Eventless transitions, wildcard events, and composable guards — relatively small implementation effort with significant capability gains.
2. **Differentiation to preserve**: Design-by-Contract, dynamic modification, and extension sandbox are genuine differentiators worth marketing and extending.
3. **Tooling gap**: Visualization is table-stakes for developer adoption. Even basic Graphviz/Mermaid diagram export would close a significant gap.
4. **SCXML alignment**: Adopting the W3C event processing algorithm would provide correctness guarantees and compatibility with the formal specification.

---

## Sources

- [XState v5 Documentation](https://stately.ai/docs/xstate)
- [XState GitHub](https://github.com/statelyai/xstate)
- [Boost.SML Documentation](https://boost-ext.github.io/sml/index.html)
- [Boost.SML GitHub](https://github.com/boost-ext/sml)
- [Boost.MSM Documentation](https://www.boost.org/doc/libs/1_64_0/libs/msm/doc/HTML/index.html)
- [Spring Statemachine Reference](https://docs.spring.io/spring-statemachine/docs/current/reference/)
- [W3C SCXML Specification](https://www.w3.org/TR/scxml/)
