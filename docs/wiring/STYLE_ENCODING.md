# Style Encoding — 16-bit Address Map for Cognitive Parameters

> Reference document for how ThinkingStyle, RungLevel, GateState,
> and consumer-supplied weights map into the 8+8 address space.
> All types referenced here exist in ladybug-rs source.

## The 8+8 address architecture

From `src/storage/cog_redis.rs`:

```
┌─────────────────────────────────────────────────────────────┐
│                 PREFIX (8-bit) : SLOT (8-bit)                │
├─────────────┬───────────────────────────────────────────────┤
│ 0x00-0x0F   │ SURFACE — 16 prefixes × 256 = 4,096 slots    │
│ 0x10-0x7F   │ FLUID   — 112 prefixes × 256 = 28,672 slots  │
│ 0x80-0xFF   │ NODES   — 128 prefixes × 256 = 32,768 slots  │
└─────────────┴───────────────────────────────────────────────┘
```

Total: 65,536 addressable slots (u16). Prefix `0x04` is NARS.

## NARS prefix (0x04) slot allocation

```
┌────────────────────────────────────────────────────────────────┐
│ 0x04:XX  NARS INFERENCE ADDRESS SPACE                          │
├──────────┬─────────────────────────────────────────────────────┤
│ CORE     │                                                     │
│ 0x04:00  │ Deduction bias              f32                     │
│ 0x04:01  │ Induction bias              f32                     │
│ 0x04:02  │ Abduction bias              f32                     │
│ 0x04:03  │ Analogy bias                f32                     │
│ 0x04:04  │ Revision bias               f32                     │
│ 0x04:05  │ confidence_modifier         f32                     │
│ 0x04:06  │ chain_depth_delta           i8                      │
│ 0x04:07  │ AtomKind::Observe weight    f32                     │
│ 0x04:08  │ AtomKind::Deduce weight     f32                     │
│ 0x04:09  │ AtomKind::Critique weight   f32                     │
│ 0x04:0A  │ AtomKind::Integrate weight  f32                     │
│ 0x04:0B  │ AtomKind::Jump weight       f32                     │
│ 0x04:0C  │ PearlMode                   u8 (0/1/2)             │
│ 0x04:0D  │ collapse.confidence_mod     f32                     │
│ 0x04:0E  │ collapse.depth_delta        i8                      │
│ 0x04:0F  │ (reserved)                                          │
├──────────┼─────────────────────────────────────────────────────┤
│ EXTENDED │ Consumer-supplied weight vectors (pass-through)     │
│ 0x04:10  │ extended_weights[0]         f32                     │
│ 0x04:11  │ extended_weights[1]         f32                     │
│  ...     │  ...                                                │
│ 0x04:33  │ extended_weights[35]        f32  — 36 NARS slots    │
│ 0x04:34  │ extended_weights[36]        f32                     │
│  ...     │  ...                                                │
│ 0x04:54  │ extended_weights[68]        f32  — 33 more slots    │
├──────────┼─────────────────────────────────────────────────────┤
│ RESERVED │                                                     │
│ 0x04:55  │ (future inference params)                           │
│  ...     │  ...                                                │
│ 0x04:FF  │ (future inference params)                           │
└──────────┴─────────────────────────────────────────────────────┘
```

### Core slots (0x04:00–0x04:0F) — substrate-managed

These 16 slots are computed by `InferenceContext::build()` from the
four input components. Ladybug reads and writes them during inference.

### Extended slots (0x04:10–0x04:54) — consumer-managed

These 69 slots are allocated for consumer weight vectors. Ladybug
stores and retrieves them but never interprets them. Two common
consumer encodings:

**36-slot NARS inference weights** (0x04:10–0x04:33):
Consumer packs per-rule biases for fine-grained inference control.
The 36 slots encode biases across 6 rule families × 6 modulation
axes. Ladybug passes these through to consumer-registered hooks.

**33-slot cognitive fingerprint** (0x04:34–0x04:54):
Consumer packs a multidimensional cognitive profile. Typical layout:
3 Pearl dimensions + 9 Rung dimensions + 5 Sigma dimensions +
8 Operation dimensions + 8 spare. Ladybug does not interpret these.

### Why extended weights are pass-through

Ladybug is a substrate. It defines 12 ThinkingStyles and their
mechanical effects on inference (the core 16 slots). But consumers
may have richer style models — more dimensions, different taxonomies,
domain-specific biases. The extended slots let consumers carry their
own encodings through the inference pipeline without ladybug needing
to understand them.

## ThinkingStyle → StyleWeights conversion

The 12 styles already have `FieldModulation` parameters in style.rs.
The conversion to StyleWeights is a projection:

```
ThinkingStyle::field_modulation()
    │
    ├── depth_bias ──────────→ Deduction bias = depth_bias × 1.0
    │                          chain_depth_delta = (depth_bias × 4) as i8
    │
    ├── breadth_bias ────────→ Induction bias = breadth_bias × 0.8
    │                          Analogy bias  = breadth_bias × 0.9
    │
    ├── noise_tolerance ────→ Abduction bias = noise_tolerance × 1.5
    │
    ├── exploration ────────→ Analogy bias  += exploration × 0.5
    │                          Abduction bias += exploration × 0.3
    │
    ├── resonance_threshold → confidence_modifier = resonance_threshold × 1.8
    │
    └── speed_bias ─────────→ chain_depth_delta -= (speed_bias × 3) as i8
```

Revision bias is always `1.0 - max(depth_bias, breadth_bias)` — it
activates when neither deep nor broad reasoning dominates.

### Worked example: Analytical style

```
FieldModulation {
    resonance_threshold: 0.85,
    fan_out: 3,
    depth_bias: 1.0,
    breadth_bias: 0.1,
    noise_tolerance: 0.05,
    speed_bias: 0.1,
    exploration: 0.05,
}

→ StyleWeights {
    rule_biases: [
        (Deduction,  1.0),   // depth_bias × 1.0
        (Induction,  0.08),  // breadth_bias × 0.8
        (Abduction,  0.09),  // noise_tolerance × 1.5 + exploration × 0.3
        (Analogy,    0.12),  // breadth_bias × 0.9 + exploration × 0.5
        (Revision,   0.0),   // 1.0 - max(1.0, 0.1) = 0.0
    ],
    confidence_modifier: 1.53,  // 0.85 × 1.8
    chain_depth_delta: 4,       // (1.0 × 4) - (0.1 × 3) = 3.7 → 4
    extended_weights: None,     // substrate-only, no consumer extensions
}
```

### Worked example: Creative style

```
FieldModulation {
    resonance_threshold: 0.35,
    fan_out: 12,
    depth_bias: 0.2,
    breadth_bias: 1.0,
    noise_tolerance: 0.4,
    speed_bias: 0.5,
    exploration: 0.8,
}

→ StyleWeights {
    rule_biases: [
        (Deduction,  0.2),   // depth_bias × 1.0
        (Induction,  0.80),  // breadth_bias × 0.8
        (Abduction,  0.84),  // noise_tolerance × 1.5 + exploration × 0.3
        (Analogy,    1.30),  // breadth_bias × 0.9 + exploration × 0.5
        (Revision,   0.0),   // 1.0 - max(0.2, 1.0) = 0.0
    ],
    confidence_modifier: 0.63,  // 0.35 × 1.8
    chain_depth_delta: -1,      // (0.2 × 4) - (0.5 × 3) = -0.7 → -1
    extended_weights: None,
}
```

## All 12 styles — quick reference

| Style | Dom. rule | Conf. | Depth | Character |
|---|---|---|---|---|
| Analytical    | Deduction   | 1.53 | +4 | Deep, precise, narrow |
| Convergent    | Deduction   | 1.35 | +2 | Structured, moderate |
| Systematic    | Deduction   | 1.26 | +2 | Methodical, thorough |
| Creative      | Analogy     | 0.63 | -1 | Broad, noisy, novel |
| Divergent     | Analogy     | 0.72 | -1 | Wide, associative |
| Exploratory   | Abduction   | 0.54 | -1 | Uncertain, maximal search |
| Focused       | Deduction   | 1.62 | +4 | Extreme depth, no breadth |
| Diffuse       | Induction   | 0.81 | -1 | Soft, spread, pattern |
| Peripheral    | Abduction   | 0.36 | -2 | Edge detection, noise-heavy |
| Intuitive     | Analogy     | 0.90 | -2 | Fast, shallow, gut-feel |
| Deliberate    | Deduction   | 1.26 | +2 | Slow, careful, balanced |
| Metacognitive | Revision    | 0.90 |  0 | Self-monitoring, neutral |

## RungLevel → AtomGate encoding

From rung.rs, the 10 levels fall into 4 bands:

```
Band         Levels                  Dominant AtomKind
─────────    ───────────────────     ──────────────────
Surface      0-Surface               Observe (1.5)
             1-Shallow
             2-Contextual

Analogical   3-Analogical            Jump (1.5)
             4-Abstract
             5-Structural

Meta         6-Counterfactual        Critique (1.5)
             7-Meta

Recursive    8-Recursive             (none — all 1.0)
             9-Transcendent
```

AtomGate stores 5 weights, one per AtomKind. The `from_rung()`
constructor selects weights based on the rung's band.

## GateState → CollapseModulation encoding

From collapse_gate.rs:

```
GateState    SD range       confidence_mod    depth_delta
─────────    ──────────     ──────────────    ───────────
Flow         < 0.15         1.4               -2
Hold         0.15–0.35      1.0                0
Block        > 0.35         0.6               +2
```

## Combined example

Consumer requests inference with:
- ThinkingStyle: Exploratory
- RungLevel: Meta (7)
- GateState: Block

```
StyleWeights:
  Deduction=0.4, Induction=0.64, Abduction=1.09, Analogy=1.12, Revision=0.2
  confidence=0.54, depth=-1

AtomGate (from Rung 7 / Meta band):
  Observe=0.5, Deduce=0.5, Critique=1.5, Integrate=1.0, Jump=1.0

CollapseModulation (from Block):
  confidence=0.6, depth=+2

PearlMode: Imagine (Counterfact)

InferenceContext::build() computes:
  min_confidence = 0.54 × 0.6 = 0.324
  max_chain_depth = BASE(5) + (-1) + (+2) = 6
```

This context tells the inference engine: favor abductive leaps,
gate hard toward critique, use counterfactual reasoning mode,
accept low-confidence results, and search moderately deep.

The Exploratory style's high noise tolerance meets the Meta rung's
critique emphasis meets the Block state's deep-exploration mandate.
The result is maximally exploratory inference — exactly what you want
when the system is uncertain and needs to question its own reasoning.
