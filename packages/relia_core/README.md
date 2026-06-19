# `@relia_core` Python Package Wrapper

This folder marks the product-facing ReliaGuard core package boundary. The current implementation keeps `src/reliaguard_studio` as the source of truth to preserve backward-compatible tests and CLI commands. Product apps import the same engine through the installed Python package.

Future extraction path:

1. Move reliance-state schemas and scoring services into `packages/relia_core/src/relia_core`.
2. Keep dataset adapters and paper-specific analysis under the research package.
3. Publish `relia_core` as the deployable inference/evaluation library.
