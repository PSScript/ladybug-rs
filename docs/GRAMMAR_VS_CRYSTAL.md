# Grammar Triangle vs Context Crystal

## The Question

Is the `grammar/` Grammar Triangle approach better, or the `extensions/context_crystal` SPO+Qualia approach?

---

## Grammar Triangle (Static Analysis)

```
Input: Single utterance "I want to understand this deeply"

         CAUSALITY
            /\
           /  \
          /    \
   NSM <──⊕──> QUALIA
 (65 primes)   (18D)
         │
         ↓
    10K FINGERPRINT
```

**Pros:**
- Simple: one utterance → one fingerprint
- Grounded in linguistics (Wierzbicka's NSM primitives)
- No context needed
- Fast: O(text_length)

**Cons:**
- **Loses temporal context** — "I want" means different things depending on what came before
- Static snapshot — meaning is frozen, not flowing
- No anticipation — doesn't encode what comes next
- Keyword-based — crude approximation of semantic analysis

---

## Context Crystal (Dynamic Flow)

```
Input: Window of 5 sentences around the utterance

   [S-2] → [S-1] → [S0] → [S+1] → [S+2]
     │       │      │       │       │
     ▼       ▼      ▼       ▼       ▼
  ┌─────────────────────────────────────┐
  │           5×5×5 Crystal             │
  │                                     │
  │   t=0   t=1   t=2   t=3   t=4      │ ← Temporal axis
  │    │     │     │     │     │       │
  │  ┌───┬───┬───┬───┬───┐            │
  │  │   │   │ ★ │   │   │ ← S0 lives here
  │  └───┴───┴───┴───┴───┘            │
  │     S axis × O axis                │
  └─────────────────────────────────────┘
```

**Pros:**
- **Meaning is flow** — the crystal captures how meaning evolves
- **Mexican hat in time** — emphasize present, de-emphasize distant past/future
- **Anticipation** — S+1, S+2 encode what comes next (predictive)
- **SPO is richer than NSM** — actual semantic roles, not just keyword weights
- **Qualia already included** — felt-sense is first-class
- **O(1) resonance lookup** — same as SPO crystal

**Cons:**
- Requires context (5 sentences)
- More complex to populate
- Needs good SPO extraction (parsing)
- 5×5×5 = 125 cells = larger memory footprint

---

## The Insight: NSM is Already in SPO

The 65 NSM primitives map to SPO naturally:

| NSM Primitive | SPO Encoding |
|---------------|--------------|
| I, YOU, SOMEONE | Subject axis |
| THINK, KNOW, WANT, FEEL | Predicate axis (mental verbs) |
| SOMETHING, BODY | Object axis |
| GOOD, BAD | Qualia.valence |
| BEFORE, AFTER, NOW | Temporal position in crystal |
| BECAUSE, IF | Causality encoded in S-2→S-1→S0 flow |

**The Grammar Triangle is a special case of the Context Crystal where:**
- Context window = 1 sentence
- S axis = collapsed (no subject distinction)
- O axis = collapsed (no object distinction)
- Only t=2 (center) is populated

---

## The Real Question: What Are We Optimizing For?

### Use Grammar Triangle when:
- You have isolated utterances (chat messages, tweets)
- Speed matters more than context
- You want linguistic interpretability
- Context is unavailable or unreliable

### Use Context Crystal when:
- You have flowing discourse (conversations, narratives)
- Meaning depends on what came before
- You want to capture emotional arcs
- You need to predict what comes next

---

## Hybrid Approach: Both

The best system uses **both**:

```rust
pub struct MeaningExtractor {
    /// For quick single-utterance analysis
    grammar: GrammarTriangle,
    
    /// For rich contextual embedding
    crystal: ContextCrystal,
}

impl MeaningExtractor {
    /// Quick fingerprint (no context needed)
    pub fn quick(&self, text: &str) -> Fingerprint {
        self.grammar.from_text(text).to_fingerprint()
    }
    
    /// Rich contextual fingerprint
    pub fn contextual(&self, context: &[String]) -> Fingerprint {
        let atoms: Vec<_> = context.iter()
            .enumerate()
            .map(|(i, s)| self.parse_sentence(s, i as i32 - 2))
            .collect();
        
        self.crystal.insert_context(&atoms);
        self.crystal.mexican_hat_resonance(1.0, 0.3)
    }
}
```

---

## Recommendation

For ladybug-rs as a **universal cognitive substrate**:

1. **Keep Grammar Triangle** — it's the fast path, useful for indexing
2. **Add Context Crystal** — it's the rich path, useful for understanding
3. **Make them interoperable** — both produce 10K fingerprints
4. **Use Grammar Triangle for ingestion** — fast, no context needed
5. **Use Context Crystal for query** — when you have surrounding context

The fingerprints should be compatible:
- `GrammarTriangle::to_fingerprint()` → 10K bits
- `ContextCrystal::central_resonance()` → 10K bits
- Both can be compared with Hamming distance
- The Crystal's central slice should correlate with Grammar Triangle

---

## Implementation Priority

1. ✅ Grammar Triangle (done) — fast ingestion path
2. ✅ Context Crystal (done) — rich query path
3. 🔜 SPO extraction — parse sentences into Subject/Predicate/Object
4. 🔜 Integration — unified API that chooses the right approach
5. 🔜 Jina embeddings — replace keyword heuristics with real vectors

The missing piece is **SPO extraction from text**. Currently both use:
- Grammar Triangle: keyword heuristics
- Context Crystal: requires pre-parsed SPO

We need: `fn parse_spo(text: &str) -> (Fingerprint, Fingerprint, Fingerprint)`

Options:
- Dependency parsing (spacy, stanza)
- LLM extraction (expensive)
- Rule-based (fast but brittle)
- Jina embeddings + role binding (what SPO crystal already does)

---

## Conclusion

**The Context Crystal is more powerful** because meaning is fundamentally contextual.

**But Grammar Triangle is more practical** for now because it doesn't require context.

**The right answer is both** — Grammar Triangle for fast ingestion, Context Crystal for rich understanding. They're not competing approaches; they're complementary tools for different situations.

The real innovation is: **both produce 10K fingerprints that can resonate with each other**.
