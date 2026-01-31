# Ladybug-rs Wiring — Claude Code Orchestrator Prompt

You are working on **ladybug-rs**, an agnostic cognitive substrate in Rust.
This is NOT a personality. This is NOT Ada. This is a general-purpose
cognitive architecture that any system could use.

## Critical rule: Chinese wall

**DO NOT modify any existing `.rs` files.**

You may ONLY:
1. Create NEW files (like `src/nars/context.rs`)
2. Add `pub mod` lines to existing `mod.rs` files (one line, append only)
3. Add `pub use` re-exports to existing `mod.rs` files (append only)
4. Add files under `docs/`

If a wiring instruction says "connect X to Y", it means:
- Create a new file that imports from X and Y
- Add the new module to the relevant mod.rs
- NEVER change the internals of X or Y

## Repository context

- Repository: `AdaWorldAPI/ladybug-rs` (branch: `main`)
- Wiring plan: `docs/wiring/FOUR_WIRES.md`
- Current code: ~18,800 lines across 65 files, 141 tests passing

## Your ONE job right now

Create `src/nars/context.rs` as described in `docs/wiring/FOUR_WIRES.md`.

This file defines:
- `InferenceContext` — the struct every inference call will require
- `StyleWeights` — biases for inference rules (consumer fills these)
- `AtomGate` — rung-derived gating of atom kinds
- `PearlMode` — SEE/DO/IMAGINE selection
- `CollapseModulation` — FLOW/HOLD/BLOCK → confidence/depth
- `InferenceRuleKind` — enum matching existing InferenceRule implementors
- `AtomKind` — atom types that can be gated

Then add to `src/nars/mod.rs` (APPEND ONLY):
```rust
mod context;
pub use context::{InferenceContext, StyleWeights, AtomGate, PearlMode, CollapseModulation, InferenceRuleKind, AtomKind};
```

## What to import from existing code

You may `use` these existing types:
- `crate::cognitive::collapse_gate::GateState` — for CollapseModulation
- `crate::cognitive::rung::RungLevel` — for AtomGate::from_rung()
- `crate::search::causal::QueryMode` — for PearlMode::to_query_mode()
- `crate::nars::TruthValue` — if needed for context operations

DO NOT import anything from outside this repo.

## Implementation notes

- Methods marked `todo!()` in the wiring doc should have real implementations.
  The logic is simple arithmetic — the doc describes exactly what each modifier does.
- Add `Default` impl for `InferenceContext` that produces neutral (unbiased) context.
- Add `#[derive(Clone, Debug)]` on all types.
- Add doc comments explaining each field's purpose and who fills it.
- Add unit tests:
  - `test_neutral_context` — default has no biasing
  - `test_atom_gate_surface_rung` — Rung 0 → Observe boosted
  - `test_atom_gate_meta_rung` — Rung 7 → Critique boosted
  - `test_collapse_flow` — Flow → 1.4× confidence, -2 depth
  - `test_collapse_block` — Block → 0.6× confidence, +2 depth
  - `test_build_stacks_modifiers` — style × collapse = combined

## Verification

```bash
cargo check          # must pass
cargo test           # all 141 existing + new tests pass
cargo clippy         # zero warnings
```

## What comes AFTER (not your job now)

Future PRs will:
1. Add a `Reasoner` trait in `src/nars/reasoner.rs` that requires `&InferenceContext`
2. Add temporal epistemology module
3. Add universal grammar module
4. Add spectroscopy module

Each of those will follow the same pattern: new file, append mod.rs,
never touch existing code.
