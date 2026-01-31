# Inference Context — Style-Driven Reasoning

> New file: `src/nars/context.rs`
> Connects: ThinkingStyle → NARS rules, RungLevel → atom gating,
> QueryMode → causal mode, GateState → confidence/depth modulation.
> No existing files are modified.

## Problem

Ladybug-rs has four independent cognitive subsystems that influence
inference but aren't wired together:

```
src/cognitive/style.rs       — 12 ThinkingStyles with 7 FieldModulation params
src/cognitive/rung.rs        — 10 RungLevels in 4 bands (Surface/Analogical/Meta/Recursive)
src/search/causal.rs         — 3 QueryModes (Correlate/Intervene/Counterfact)
src/cognitive/collapse_gate.rs — 3 GateStates (Flow/Hold/Block) with SD thresholds
```

When a consumer calls an inference rule (`Deduction::apply()`), none
of this context is available. The rule applies blindly. There is no
mechanism to say "this inference should favor abduction because we're
at Rung 7 (Meta) in a Block state with an Exploratory thinking style."

## Solution: InferenceContext

One struct that aggregates all four subsystems into a required
parameter for inference operations. Four internal components,
each derived from an existing substrate type:

```
┌──────────────────────────────────────────────────────────────┐
│                     InferenceContext                          │
├──────────────┬──────────────┬────────────┬───────────────────┤
│ StyleWeights │   AtomGate   │ PearlMode  │ CollapseModulation│
│              │              │            │                   │
│ 12 styles →  │ RungLevel →  │ QueryMode  │ GateState →       │
│ 5 rule       │ 5 atom-kind  │ selection  │ confidence ×      │
│ biases +     │ weights per  │            │ depth delta       │
│ confidence   │ rung band    │            │                   │
│ modifier +   │              │            │                   │
│ depth delta  │              │            │                   │
└──────────────┴──────────────┴────────────┴───────────────────┘
```

## Component 1: StyleWeights

Derived from `ThinkingStyle` (style.rs). Each of the 12 existing
thinking styles maps to a set of inference rule biases.

```
                    ThinkingStyle
                    ┌──────────────────┐
   Convergent       │ Analytical       │    Divergent
   Cluster          │ Convergent       │    Cluster
                    │ Systematic       │
                    ├──────────────────┤
                    │ Creative         │
                    │ Divergent        │
                    │ Exploratory      │
                    ├──────────────────┤
   Attention        │ Focused          │    Speed
   Cluster          │ Diffuse          │    Cluster
                    │ Peripheral       │
                    ├──────────────────┤
                    │ Intuitive        │    Meta
                    │ Deliberate       │    Cluster
                    │ Metacognitive    │
                    └──────────────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │   StyleWeights   │
                    │                  │
                    │ rule_biases:     │
                    │   Deduction  f32 │
                    │   Induction  f32 │
                    │   Abduction  f32 │
                    │   Analogy    f32 │
                    │   Revision   f32 │
                    │                  │
                    │ confidence: f32  │
                    │ depth_delta: i8  │
                    └──────────────────┘
```

Example mappings (positive = prefer, negative = suppress):

| ThinkingStyle | Deduction | Induction | Abduction | Analogy | Revision | confidence | depth |
|---|---|---|---|---|---|---|---|
| Analytical    | +0.8  | +0.3  | -0.2  | -0.3  | +0.5  | 1.3 | +2 |
| Creative      | -0.3  | +0.5  | +0.8  | +0.9  | +0.2  | 0.7 | -1 |
| Focused       | +0.9  | -0.2  | -0.5  | -0.4  | +0.3  | 1.5 | +3 |
| Exploratory   | -0.1  | +0.7  | +0.6  | +0.8  | +0.4  | 0.6 | -2 |
| Metacognitive | +0.4  | +0.4  | +0.4  | +0.4  | +0.9  | 1.0 |  0 |

These values come from `FieldModulation` parameters already defined
in `style.rs`. The conversion is mechanical:

- `depth_bias` → favors Deduction (high depth = deep chaining)
- `breadth_bias` → favors Induction + Analogy (wide association)
- `noise_tolerance` → favors Abduction (tolerates uncertain leaps)
- `exploration` → favors Analogy + Abduction (novel connections)
- `resonance_threshold` → maps to confidence modifier
- `speed_bias` → inverse of depth delta

### Consumer-supplied weight arrays

Some consumers pack additional style biases as float arrays.
StyleWeights accepts an optional opaque weight vector for this:

```rust
pub struct StyleWeights {
    /// Per-rule bias from ThinkingStyle.
    pub rule_biases: [(InferenceRuleKind, f32); 5],

    /// Confidence modifier (multiplied with base confidence).
    pub confidence_modifier: f32,

    /// Chain depth delta (added to base max depth).
    pub chain_depth_delta: i8,

    /// Optional consumer-supplied weight vector.
    /// Ladybug does not interpret these — it passes them through
    /// to any consumer-registered inference hooks.
    /// Typical sizes: 36 (NARS inference weights) or 33 (cognitive fingerprint).
    pub extended_weights: Option<Vec<f32>>,
}
```

The `extended_weights` field is an escape hatch. Ladybug never reads it.
Consumers provide it, and consumer-registered hooks can read it back.
This keeps the substrate agnostic about what "36 weights" or
"33-dimensional fingerprint" means — that's the consumer's vocabulary.

## Component 2: AtomGate

Derived from `RungLevel` (rung.rs). Each rung band determines which
atom kinds of reasoning are active, boosted, or suppressed.

```
Rung bands (from rung.rs):
  Surface (0-2):  literal, simple inference, contextual
  Analogical (3-5): metaphor, generalized, schema
  Meta (6-7):     counterfactual, reasoning about reasoning
  Recursive (8-9): self-referential, transcendent

Atom kinds (new enum, maps to inference families):
  Observe   → Induction    (from data)
  Deduce    → Deduction    (from rules)
  Critique  → Abduction    (from effects)
  Integrate → Revision     (consolidation)
  Jump      → Analogy      (cross-domain)
```

Mapping table:

| Rung band      | Observe | Deduce | Critique | Integrate | Jump |
|---|---|---|---|---|---|
| Surface (0-2)  | **1.5** | 1.0    | 0.5      | 0.5       | 0.3  |
| Analogical (3-5)| 0.8   | 0.8    | 0.8      | 1.0       | **1.5** |
| Meta (6-7)     | 0.5    | 0.5    | **1.5**  | 1.0       | 1.0  |
| Recursive (8-9)| 1.0    | 1.0    | 1.0      | 1.0       | 1.0  |

At Surface rungs, observation dominates — gather data first.
At Analogical rungs, cross-domain jumps dominate — find connections.
At Meta rungs, critique dominates — question the reasoning itself.
At Recursive rungs, no gating — everything is in play.

## Component 3: PearlMode

Direct mapping to existing `QueryMode` (search/causal.rs):

```
PearlMode::See     → QueryMode::Correlate     (Rung 1: association)
PearlMode::Do      → QueryMode::Intervene     (Rung 2: intervention)
PearlMode::Imagine → QueryMode::Counterfact   (Rung 3: counterfactual)
```

This is a thin wrapper, not a new concept. It exists so that
`InferenceContext` can name the causal mode without importing
search types directly into the NARS module.

## Component 4: CollapseModulation

Derived from `GateState` (collapse_gate.rs):

```
GateState::Flow  → confidence × 1.4, depth - 2  (commit fast)
GateState::Hold  → confidence × 1.0, depth + 0  (neutral)
GateState::Block → confidence × 0.6, depth + 2  (explore deep)
```

Rationale: when SD is low (Flow), the system is certain and should
commit quickly with high confidence. When SD is high (Block), the
system is uncertain and should explore more deeply before committing.
This matches the existing threshold logic in `collapse_gate.rs`
(Flow < 0.15 SD, Block > 0.35 SD).

## Stacking: how the four components combine

```rust
impl InferenceContext {
    pub fn build(
        style: StyleWeights,
        gate: AtomGate,
        mode: PearlMode,
        collapse: CollapseModulation,
    ) -> Self {
        let min_confidence =
            style.confidence_modifier * collapse.confidence_modifier;
        let max_chain_depth =
            (BASE_DEPTH as i16
             + style.chain_depth_delta as i16
             + collapse.depth_delta as i16)
            .clamp(1, 20) as u8;

        Self {
            style_weights: style,
            atom_gate: gate,
            pearl_mode: mode,
            collapse,
            min_confidence,
            max_chain_depth,
        }
    }
}
```

Modifiers multiply for confidence, add for depth. All values are
substrate-defined — consumers fill the inputs, ladybug computes
the combination.

## 16-bit address space integration

The cognitive redis address map (`src/storage/cog_redis.rs`) allocates
NARS operations at prefix `0x04`:

```
0x04:00-0x04:FF — NARS inference address space (256 slots)
```

InferenceContext slots within this prefix:

```
0x04:00-0x04:04 — 5 InferenceRuleKind biases (one per rule)
0x04:05         — confidence_modifier
0x04:06         — chain_depth_delta
0x04:07-0x04:0B — 5 AtomKind weights (one per kind)
0x04:0C         — PearlMode (0=See, 1=Do, 2=Imagine)
0x04:0D         — CollapseModulation.confidence_modifier
0x04:0E         — CollapseModulation.depth_delta
0x04:0F         — reserved

0x04:10-0x04:33 — extended_weights[0..35] (consumer space, 36 slots)
0x04:34-0x04:54 — extended_weights[36..68] (consumer space, 33 slots)
0x04:55-0x04:FF — reserved for future inference parameters
```

This gives consumers 69 float slots for their own weight encodings
without ladybug needing to know what they mean. The substrate manages
the first 16 slots (`0x04:00` through `0x04:0F`). Everything from
`0x04:10` onward is pass-through.

## File creation instructions

Create `src/nars/context.rs` with:

1. `InferenceContext` — the aggregate struct
2. `StyleWeights` — rule biases + confidence + depth + optional extended
3. `InferenceRuleKind` — enum matching inference.rs implementors
4. `AtomGate` — rung-derived atom kind weights
5. `AtomKind` — 5 atom kinds mapping to inference families
6. `PearlMode` — thin wrapper over QueryMode
7. `CollapseModulation` — gate-state-derived confidence/depth

Constructors:
- `StyleWeights::from_thinking_style(style: ThinkingStyle) → Self`
- `AtomGate::from_rung(rung: RungLevel) → Self`
- `PearlMode::to_query_mode(&self) → QueryMode`
- `CollapseModulation::from_gate(gate: GateState) → Self`
- `InferenceContext::build(style, gate, mode, collapse) → Self`
- `InferenceContext::neutral() → Self`

After creating, append to `src/nars/mod.rs`:
```rust
mod context;
pub use context::{
    InferenceContext, StyleWeights, AtomGate, PearlMode,
    CollapseModulation, InferenceRuleKind, AtomKind,
};
```

## Tests

```rust
#[test] fn test_neutral_context()        // all modifiers = 1.0 / 0
#[test] fn test_style_analytical()       // Deduction boosted, Abduction suppressed
#[test] fn test_style_creative()         // Analogy boosted, Deduction suppressed
#[test] fn test_atom_gate_surface()      // Rung 0 → Observe = 1.5
#[test] fn test_atom_gate_meta()         // Rung 7 → Critique = 1.5
#[test] fn test_collapse_flow()          // Flow → 1.4× confidence, -2 depth
#[test] fn test_collapse_block()         // Block → 0.6× confidence, +2 depth
#[test] fn test_build_stacks()           // style × collapse = combined
#[test] fn test_pearl_to_query_mode()    // See→Correlate, Do→Intervene, Imagine→Counterfact
#[test] fn test_extended_weights_passthrough()  // consumer weights survive round-trip
```

## Future work

After `context.rs` exists:

1. `src/nars/reasoner.rs` — `Reasoner` trait requiring `&InferenceContext`
2. Temporal epistemology module (temporal truth values with InferenceContext)
3. Universal grammar module (sentence parsing under style bias)
4. Spectroscopy module (multi-level analysis gated by AtomGate)

Each follows the same pattern: new file, append mod.rs, existing code untouched.
