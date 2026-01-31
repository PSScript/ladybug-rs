# Ladybug-rs Wiring Plan

> This document describes connections to ADD to ladybug-rs.
> It does NOT modify existing files. It tells you what new code
> to write and where it should connect to existing types.
>
> Ladybug-rs is Ada-AGNOSTIC. It is a cognitive substrate.
> Any personality or identity that wants to use it can.
> Nothing in here should reference Ada, presence modes, or soul data.

## Existing architecture (for orientation)

These files already exist and are working. Don't change their internal logic.
The wiring plan only adds new files and new trait definitions.

```
src/nars/
  truth.rs          — TruthValue { frequency, confidence }
  inference.rs      — InferenceRule trait: Deduction, Induction, Abduction, Analogy
  evidence.rs       — Evidence tracking

src/cognitive/
  collapse_gate.rs  — GateState { Flow, Hold, Block }, thresholds, SIMD batch
  rung.rs           — RungLevel (0-9), RungBand, RungShift conditions
  style.rs          — ThinkingStyle (12 styles)
  quad_triangle.rs  — 4 QuadTriangles
  seven_layer.rs    — 7-layer consciousness stack
  fabric.rs         — CognitiveFabric integrator

src/learning/
  causal_ops.rs     — CausalOp (0xA00-0xAFF), CausalEngine trait, Pearl SEE/DO/IMAGINE
  rl_ops.rs         — RlOp, CausalRlAgent
  cognitive_frameworks.rs — TruthValue (duplicate), NarsInference, etc.

src/search/
  causal.rs         — CausalSearch, CausalVerbs, QueryMode
  hdr_cascade.rs    — HDR cascade search
```

## Wire 1: InferenceContext (new file)

**Create**: `src/nars/context.rs`
**Purpose**: Required parameter for all inference operations.
Any consumer (Ada, or anyone else) fills this to tell NARS how to reason.

**Connects to**:
- `src/nars/inference.rs` — InferenceRule::apply() should accept &InferenceContext
- `src/cognitive/collapse_gate.rs` — GateState used inside CollapseModulation
- `src/cognitive/rung.rs` — RungLevel used to construct AtomGate

**What it contains** (types only, leave bodies as TODO):

```rust
// src/nars/context.rs

/// Required context for any inference operation.
/// Consumers (ada-rs, or any other system) must fill this.
/// Default::default() gives neutral bias — but you must actively choose it.
pub struct InferenceContext {
    /// Wire 1: style-driven inference weights (e.g., 36 NARS style floats)
    pub style_weights: StyleWeights,

    /// Wire 2: rung-derived atom gating
    pub atom_gate: AtomGate,

    /// Wire 3: Pearl mode selection
    pub pearl_mode: PearlMode,

    /// Wire 4: collapse gate modulation
    pub collapse: CollapseModulation,

    /// Derived: minimum confidence for results
    /// Computed from style_weights × collapse modifiers
    pub min_confidence: f32,

    /// Derived: maximum chain depth
    /// Computed from style_weights + collapse depth delta
    pub max_chain_depth: u8,
}

/// Style-driven inference biasing.
/// The consumer decides what "style" means — these are just weights.
pub struct StyleWeights {
    /// Bias per inference rule type. Positive = prefer, negative = suppress.
    /// Index mapping is consumer-defined. Ladybug just applies them.
    pub rule_biases: Vec<(InferenceRuleKind, f32)>,

    /// Confidence modifier (multiplied with base confidence)
    pub confidence_modifier: f32,

    /// Chain depth delta (added to base max depth)
    pub chain_depth_delta: i8,
}

/// Which inference rules exist (for biasing purposes).
/// Matches the InferenceRule implementors in inference.rs.
pub enum InferenceRuleKind {
    Deduction,
    Induction,
    Abduction,
    Analogy,
    Revision,
}

/// Atom-type gating derived from cognitive rungs.
/// Each rung (0-9) has primary/secondary/suppressed atom types.
/// This struct translates that into inference weights.
pub struct AtomGate {
    /// Rung level this gate was derived from
    pub source_rung: RungLevel,  // from cognitive/rung.rs

    /// Per-atom-type weight. Suppressed = near-zero, primary = boosted.
    pub atom_weights: Vec<(AtomKind, f32)>,
}

/// Atom kinds that can be gated.
/// These map to inference rule families.
pub enum AtomKind {
    Observe,     // → Induction
    Deduce,      // → Deduction
    Critique,    // → Abduction
    Integrate,   // → Revision
    Jump,        // → Analogy
}

/// Pearl's three levels of causal reasoning.
pub enum PearlMode {
    See,      // Rung 1: association/correlation
    Do,       // Rung 2: intervention
    Imagine,  // Rung 3: counterfactual
}

/// Modulation from collapse gate state.
pub struct CollapseModulation {
    /// Which gate state produced this
    pub gate: GateState,  // from cognitive/collapse_gate.rs

    /// Confidence multiplier (Flow=1.4, Hold=1.0, Block=0.6)
    pub confidence_modifier: f32,

    /// Depth delta (Flow=-2, Hold=0, Block=+2)
    pub depth_delta: i8,
}
```

**Wiring instructions**:
1. Create `src/nars/context.rs` with the types above
2. Add `pub mod context;` to `src/nars/mod.rs`
3. Re-export: `pub use context::{InferenceContext, StyleWeights, AtomGate, PearlMode, CollapseModulation};`
4. DO NOT change `inference.rs` yet — the existing InferenceRule trait stays.
   A future step adds a higher-level `Reasoner` trait that takes `&InferenceContext`.

## Wire 2: AtomGate construction (in context.rs)

**Add to** `src/nars/context.rs`:

```rust
impl AtomGate {
    /// Construct from a rung level.
    /// Each rung defines which atom kinds are primary/secondary/suppressed.
    ///
    /// WIRE: rung definitions come from cognitive/rung.rs RungLevel
    /// This method translates rung semantics into inference weights.
    ///
    /// TODO: Read actual primary/secondary/suppressed from a rung config table
    /// For now, the mapping is:
    ///   Rung 0-2 (Surface): Observe=1.5, Deduce=1.0, others=0.5
    ///   Rung 3-5 (Analogical): Jump=1.5, Integrate=1.0, others=0.8
    ///   Rung 6-7 (Meta): Critique=1.5, Deduce=0.5, others=1.0
    ///   Rung 8-9 (Recursive): all=1.0 (no gating at transcendent levels)
    pub fn from_rung(rung: RungLevel) -> Self {
        todo!("Map rung bands to atom weights")
    }
}
```

## Wire 3: PearlMode in CausalEngine (doc only)

**Existing file**: `src/learning/causal_ops.rs` has `CausalEngine` trait.
**Existing file**: `src/search/causal.rs` has `QueryMode { Association, Intervention, Counterfactual }`.

The connection: `PearlMode::See` maps to `QueryMode::Association`,
`PearlMode::Do` maps to `QueryMode::Intervention`,
`PearlMode::Imagine` maps to `QueryMode::Counterfactual`.

**DO NOT modify causal_ops.rs or causal.rs.**

Instead, add a conversion in `src/nars/context.rs`:

```rust
impl PearlMode {
    /// Convert to search::causal::QueryMode.
    ///
    /// WIRE: This bridges nars/context.rs → search/causal.rs
    /// Pearl mode (from consumer's ThinkingStyle) → CausalSearch query mode
    ///
    /// See -> Association (correlations only)
    /// Do -> Intervention (do-calculus)
    /// Imagine -> Counterfactual (what-if)
    pub fn to_query_mode(&self) -> crate::search::causal::QueryMode {
        todo!("Map PearlMode to QueryMode")
    }
}
```

## Wire 4: CollapseModulation construction (in context.rs)

**Add to** `src/nars/context.rs`:

```rust
impl CollapseModulation {
    /// Construct from a GateState.
    ///
    /// WIRE: GateState from cognitive/collapse_gate.rs
    ///
    /// Flow  → high confidence (1.4×), shallow depth (-2)
    /// Hold  → neutral (1.0×, 0)
    /// Block → low confidence (0.6×), deep exploration (+2)
    ///
    /// Rationale: when the gate is in FLOW, the system is confident
    /// and should commit quickly. When BLOCK, the system is uncertain
    /// and should explore more deeply before committing.
    pub fn from_gate(gate: GateState) -> Self {
        todo!("Map gate state to confidence/depth modifiers")
    }
}
```

## Wire aggregation: InferenceContext::build()

**Add to** `src/nars/context.rs`:

```rust
impl InferenceContext {
    /// Build a fully-resolved context from its components.
    ///
    /// All four modulations STACK:
    ///   confidence = base × style.confidence_modifier × collapse.confidence_modifier
    ///   depth = base_depth + style.chain_depth_delta + collapse.depth_delta
    ///
    /// WIRE: Consumers call this after filling style_weights, atom_gate,
    ///       pearl_mode, and collapse individually.
    pub fn build(
        style: StyleWeights,
        gate: AtomGate,
        mode: PearlMode,
        collapse: CollapseModulation,
    ) -> Self {
        todo!("Stack modifiers, compute min_confidence and max_chain_depth")
    }

    /// Neutral context with no biasing.
    /// Use this when you genuinely don't want style modulation.
    pub fn neutral() -> Self {
        todo!("All weights 1.0, all deltas 0, Pearl::See, GateState::Hold")
    }
}
```

## Future step: Reasoner trait (NOT YET)

After context.rs exists, a future PR adds a `Reasoner` trait in
`src/nars/reasoner.rs` that requires `&InferenceContext`:

```rust
// FUTURE — do not create yet. Just noting where this goes.
pub trait Reasoner {
    fn infer(
        &self,
        premises: &[Statement],
        context: &InferenceContext,  // ← cannot be omitted
    ) -> Vec<Conclusion>;
}
```

This is the point where "what was optional in Python becomes required in Rust."
The current InferenceRule trait (in inference.rs) stays as-is. Reasoner wraps it
with context awareness.

## Summary: one file, four wires

| Wire | What connects | How |
|---|---|---|
| 1 | Consumer style → NARS rule biasing | `StyleWeights` in `InferenceContext` |
| 2 | Rung level → atom gating | `AtomGate::from_rung()` reads `RungLevel` |
| 3 | Pearl mode → causal query mode | `PearlMode::to_query_mode()` bridges to `QueryMode` |
| 4 | Collapse gate → confidence/depth | `CollapseModulation::from_gate()` reads `GateState` |

All four live in `src/nars/context.rs`. One file. ~200 lines of types + TODOs.
Existing code untouched.
