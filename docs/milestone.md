# Milestone: Causal Interpretability for Free

**Date:** 2026-01-31
**Session:** Crystal feature test fixes → Architectural epiphany

---

## The Bug That Revealed Everything

While fixing crystal feature tests (183 now passing), we discovered a fundamental confusion in the codebase: **correlation and causation were being conflated**.

The `CorrelationStore::query()` was using HDR cascade (similarity search) to answer causal queries. This is a category error — the computational equivalent of "correlation does not imply causation."

```rust
// WRONG: Using similarity to find causal edges
let matches = self.index.search(&pattern, k);  // HDR cascade

// RIGHT: Traversing the causal graph
let edges = self.edges_from(node);  // Graph walk
```

---

## The Separation

```
SEARCH (HDR cascade)         WALK (graph traversal)
─────────────────────        ─────────────────────
"What resonates?"            "What follows?"
Similarity metric            Edge following
O(n) candidates              O(1) per hop
Returns: similar FPs         Returns: connected nodes
Use: intuition, qualia       Use: causation, reasoning
```

Pearl's three rungs require three different structures:

| Rung | Question | Structure | Operation |
|------|----------|-----------|-----------|
| 1: SEE | "What correlates?" | HDR Cascade | Similarity search |
| 2: DO | "What causes?" | Causal Graph | Edge traversal |
| 3: IMAGINE | "What if?" | World Model | Counterfactual query |

---

## The Epiphany: Causal Interpretability for Free

### Current State of ML

```
Training: X → Y, loss, done
Post-hoc: "Why did it do that?" ← years of research, still unsolved
          "What if...?" ← can't answer without retraining
```

We keep: **weights** (correlation summary)
We discard: **the entire causal history of how they got there**

Then we spend years trying to reconstruct what we discarded:
- SHAP values
- Attention maps
- Gradient attribution
- Integrated gradients
- Concept activation vectors
- Mechanistic interpretability

All asking: *"Please, neural net, tell us what you learned and why."*

The net can't answer. **Because we didn't log the causation.**

### With Ladybug Logging

Every gradient step IS an intervention. We just don't save it.

```rust
fn training_step(&mut self, batch: &Batch) -> Loss {
    let output = self.forward(batch);
    let loss = self.loss(output, batch.labels);
    self.backward(loss);

    // === LADYBUG LOGGING ===

    // Rung 1: Correlations (what appeared together)
    self.cascade.insert(batch.fingerprint());

    // Rung 2: Causations (what caused what)
    for (layer, grad) in self.gradients() {
        if grad.magnitude() > SIGNIFICANT {
            self.causal.add_edge(
                layer.addr(),      // source
                output.addr(),     // target
                VERB_CAUSED,
                grad.magnitude()
            );
        }
    }

    // Rung 3: Counterfactuals (for "what if" queries)
    self.world.snapshot(self.state());

    loss
}
```

### After Training

```rust
// "Why does this model see cats?"
model.causal.traverse_to(&cat_neuron, depth: 10);
// Returns: training_image_4521 ──▶ edge_detector ──▶ ear_shape ──▶ cat_neuron

// "What if I'd used different data?"
model.world.imagine(do(remove_training_image_4521));
// Returns: cat_accuracy -= 3%

// "What's most important?"
model.causal.highest_influence(&output_layer);
// Returns: ranked list of actual causal contributors
```

**The causal graph IS the explanation. You don't interpret it. You read it.**

---

## Why 8+8 Addressing Matters

The causal graph could be enormous (millions of training steps × thousands of neurons). But with 8+8 addressing:

```
rain lives at           0x8042  (node zone)
rain's CAUSES edges at  0x1042  (fluid zone = 0x10 + node suffix)
                          │
                          ├─▶ wet_ground (0x8107)
                          │       └─▶ mud (0x8203)
                          └─▶ humidity (0x8089)
                                  └─▶ fog
```

- O(1) edge lookup per node
- O(1) logging per training step
- O(depth) for any causal query

The addressing scheme that seemed like premature optimization is actually **the enabler** for tractable causal logging.

---

## What This Means

1. **Interpretability becomes a side effect**, not a research problem
2. **Debugging becomes graph traversal**, not guesswork
3. **Transfer learning has principled foundations** — causal structure transfers, correlations don't
4. **Robustness improves** — causal models handle distribution shift
5. **Explanations are exact**, not statistical approximations

---

## Next Steps

1. **Clean separation in code**: `HdrCascade`, `CausalGraph`, `WorldModel` as distinct structures
2. **Harvest traversal API from PR #24**: `match_edges_from`, `match_edges_to`, `all_edges_from`
3. **Prototype training logger**: Hook into PyTorch/JAX gradient tape
4. **Benchmark overhead**: Verify O(1) logging doesn't slow training significantly

---

## The Lesson

The failing tests weren't just bugs. They were the architecture telling us it needs to be **three things, not one**.

Correlation. Causation. Counterfactual.

Three questions. Three structures. Three query types.

No mixing.

---

*"The causal graph IS the explanation. You don't interpret it. You read it."*

🦋
