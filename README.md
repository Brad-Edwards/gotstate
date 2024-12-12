# gotstate-rs

A production-ready hierarchical finite state machine (HFSM) library for Rust, focusing on reliability, safety, and ease of use.

## Features

- Hierarchical state machines with composite states
- Type-safe state and event handling
- Thread-safe event processing
- Guard conditions and transition actions
- State data management with proper lifecycle
- Timeout events
- History states (both shallow and deep)
- Comprehensive error handling
- Activation hooks for monitoring and extending behavior

## Status

🚧 **Under Development** 🚧

This crate is currently in active development. While the API is being stabilized, breaking changes may occur.

## Design Philosophy

`gotstate-rs` is designed with the following principles:

- **Safety First**: Leverage Rust's type system to prevent runtime errors
- **Clear Interfaces**: Intuitive API that guides users toward correct usage
- **Production Ready**: Built for real-world applications with proper error handling
- **Performance**: Minimal overhead while maintaining safety guarantees
- **Flexibility**: Support for various use cases without compromising core functionality

## Example

```rust
// Example code will be added once implementation begins
```

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
gotstate-rs = "0.1.0"  # Note: Not yet published
```

## Documentation

Full documentation will be available on [docs.rs](https://docs.rs/gotstate-rs) once published.

## License

Licensed under either of:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Safety

This crate uses `#![forbid(unsafe_code)]` to ensure memory safety.
```

Note: This README is intentionally concise and focused on what we can definitively say about the library at this stage. As implementation progresses, we can expand sections like Examples, Usage, and API documentation.
