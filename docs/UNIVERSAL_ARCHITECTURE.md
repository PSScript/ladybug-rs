# Universal Architecture Roadmap

**Status**: What exists vs what's needed for truly universal cognitive substrate.

---

## Current State

```
┌─────────────────────────────────────────────────────────────────┐
│                    LADYBUGDB TODAY                               │
│                                                                  │
│   ✅ SQL (DataFusion)                                           │
│   ✅ Cypher (transpiled to recursive CTEs)                      │
│   ✅ Vector ANN (Lance)                                         │
│   ✅ Hamming similarity (AVX-512/AVX2/NEON, 65M cmp/sec)       │
│   ✅ VSA operations (bind/bundle/sequence)                      │
│   ✅ NARS inference (deduction/revision/abduction)              │
│   ✅ Counterfactual worlds (fork/what_if)                       │
│   ✅ Learning loop (moment/session/blackboard)                  │
│   ✅ Collapse gates (FLOW/HOLD/BLOCK)                           │
│   ✅ Lance columnar storage (Arrow-native)                      │
│                                                                  │
│   Extensions:                                                    │
│   ✅ Crystal Savant (codebook)                                  │
│   ✅ Crystal Memory (hologram)                                  │
│   ✅ SPO Crystal (3D graph)                                     │
│   ✅ Crystal Compress                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**What no other system has**: All of the above unified, at SIMD speed, on columnar storage, with cognitive operations as first-class primitives.

---

## What's Missing for Universal

### 1. Grammar Layer — The Missing Foundation

You have SPO Crystal (Subject-Predicate-Object), but no *parsing layer* that turns arbitrary input into SPO triples. LangExtract is listed but not implemented.

```rust
// MISSING: Universal Grammar Parser
pub trait GrammarParser {
    /// Any input → SPO triples + qualia annotations
    fn parse(&self, input: &[u8]) -> Vec<AnnotatedTriple>;
}

// Should handle:
// - Natural language → NSM decomposition → SPO
// - JSON/XML → structure → SPO
// - Code AST → dependency → SPO  
// - Images (via CLIP) → description → SPO
// - Audio (via Whisper) → transcript → SPO
```

Without this, you need external APIs for every modality. With it, everything becomes SPO + qualia, stored uniformly.

**Location**: `src/grammar/`

**Dependencies**: `tree-sitter`, NSM lexicon, construction grammar rules

---

### 2. Modality Adapters — Input Universality

```rust
// MISSING: Modality tower
pub enum Input {
    Text(String),
    Json(serde_json::Value),
    Image(Vec<u8>),      // → local CLIP → fingerprint
    Audio(Vec<u8>),      // → local Whisper → text → fingerprint
    Code(String, Lang),  // → tree-sitter → AST → fingerprint
    Binary(Vec<u8>),     // → SimHash/MinHash → fingerprint
}

impl Input {
    pub fn to_fingerprint(&self) -> Fingerprint {
        match self {
            Self::Text(t) => Fingerprint::from_text(t),
            Self::Json(j) => Fingerprint::from_json(j),
            Self::Image(bytes) => local_clip_embed(bytes).project_hamming(),
            Self::Audio(bytes) => local_whisper(bytes).to_fingerprint(),
            Self::Code(src, lang) => treesitter_fingerprint(src, lang),
            Self::Binary(bytes) => simhash_fingerprint(bytes),
        }
    }
}
```

Right now you're dependent on Jina for embeddings. For truly universal, need local paths:

| Modality | Local Path | API Fallback |
|----------|------------|--------------|
| Text | SimHash / MinHash | Jina |
| Code | Tree-sitter AST | - |
| Images | Candle CLIP | OpenAI CLIP |
| Audio | Whisper.cpp | OpenAI Whisper |
| Binary | SimHash | - |

**Location**: `src/modality/`

**Dependencies**: `candle`, `whisper-rs`, `tree-sitter`, `simhash`

---

### 3. Projection Operators — From Any Embedding Space

```rust
// MISSING: Universal projection
pub trait ProjectToHamming {
    fn project(&self, embedding: &[f32]) -> Fingerprint;
}

pub struct LSHProjector {
    planes: Vec<Vec<f32>>,  // Random hyperplanes
}

pub struct LearnedProjector {
    matrix: ndarray::Array2<f32>,  // Trained on corpus
}

pub struct IterativeProjector {
    target_similarity: f32,
    max_iterations: usize,
}

impl LSHProjector {
    pub fn new(input_dim: usize, output_bits: usize) -> Self {
        // Generate random hyperplanes
        let mut rng = rand::thread_rng();
        let planes = (0..output_bits)
            .map(|_| (0..input_dim).map(|_| rng.gen::<f32>() - 0.5).collect())
            .collect();
        Self { planes }
    }
    
    pub fn project(&self, embedding: &[f32]) -> Fingerprint {
        let mut fp = Fingerprint::zero();
        for (i, plane) in self.planes.iter().enumerate() {
            let dot: f32 = embedding.iter().zip(plane).map(|(a, b)| a * b).sum();
            if dot > 0.0 {
                fp.set_bit(i, true);
            }
        }
        fp
    }
}
```

Currently Jina gives you 1024D. OpenAI gives you 1536D. Cohere gives you 1024D. Need a universal projector that takes *any* dense embedding and maps to 10K Hamming space while preserving similarity.

**Location**: `src/core/projection.rs`

**Dependencies**: `ndarray`, `rand`

---

### 4. Temporal Layer — Time as First-Class

```rust
// PARTIALLY PRESENT but not unified
pub struct TemporalFingerprint {
    pub content: Fingerprint,
    pub timestamp: u64,
    pub decay_rate: f32,       // How fast relevance fades
    pub revival_threshold: f32, // When to resurface
}

impl TemporalFingerprint {
    /// Similarity weighted by recency
    pub fn temporal_resonance(&self, other: &Self, now: u64) -> f32 {
        let content_sim = self.content.similarity(&other.content);
        let age_seconds = (now - other.timestamp) as f32 / 1000.0;
        let decay = (-age_seconds * self.decay_rate).exp();
        content_sim * decay
    }
    
    /// Mexican hat in time: not too recent, not too stale
    pub fn temporal_sweet_spot(&self, other: &Self, now: u64, center_age: f32, width: f32) -> f32 {
        let content_sim = self.content.similarity(&other.content);
        let age = (now - other.timestamp) as f32 / 1000.0;
        let x = (age - center_age) / width;
        let temporal_weight = (1.0 - x * x) * (-x * x / 2.0).exp();
        content_sim * temporal_weight.max(0.0)
    }
}

pub struct TemporalResonanceEngine {
    fingerprints: Vec<TemporalFingerprint>,
    decay_rate: f32,
    sweet_spot_center: f32,  // e.g., 1 hour ago
    sweet_spot_width: f32,   // e.g., 30 minutes
}

impl TemporalResonanceEngine {
    pub fn find_temporally_relevant(
        &self,
        query: &Fingerprint,
        now: u64,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let mut results: Vec<_> = self.fingerprints.iter()
            .enumerate()
            .map(|(i, tfp)| {
                let score = tfp.temporal_sweet_spot(
                    &TemporalFingerprint { content: query.clone(), timestamp: now, decay_rate: 0.0, revival_threshold: 0.0 },
                    now,
                    self.sweet_spot_center,
                    self.sweet_spot_width,
                );
                (i, score)
            })
            .filter(|(_, score)| *score >= threshold)
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
}
```

Mexican hat finds the sweet spot in *content space*. But there's also a temporal sweet spot — not too recent (just saw it), not too old (stale). Need time-weighted resonance.

**Location**: `src/temporal/`

**Dependencies**: None (pure Rust)

---

### 5. Federation — Multi-Instance Consensus

```rust
// MISSING: Distributed coherence
pub struct FederationClient {
    peers: Vec<PeerAddress>,
    local_id: String,
    crdt_state: CRDTState,
}

pub trait Federation {
    /// Sync moments across instances
    fn sync(&self, peer: &Database) -> SyncResult;
    
    /// Merge ice-caked decisions
    fn merge_ice_cake(&self, theirs: &IceCakedDecision) -> ConflictResolution;
    
    /// Resonance across federated corpus
    fn federated_resonate(&self, query: &Fingerprint, threshold: f32) -> Vec<FederatedMatch>;
}

pub struct FederatedMatch {
    pub moment_id: String,
    pub peer_id: String,
    pub resonance: f32,
    pub is_local: bool,
}

pub enum ConflictResolution {
    KeepLocal,
    AcceptRemote,
    Merge(IceCakedDecision),  // New decision combining both
    Conflict(String),          // Unresolvable, needs human input
}

impl FederationClient {
    /// CRDT-based moment merge
    pub fn merge_moments(&mut self, remote_moments: &[Moment]) -> Vec<MergeResult> {
        remote_moments.iter().map(|m| {
            if let Some(local) = self.find_by_fingerprint(&m.fingerprint) {
                // Same content: merge qualia (take max satisfaction, etc.)
                MergeResult::Merged(self.merge_qualia(local, m))
            } else {
                // New content: add
                MergeResult::Added(m.clone())
            }
        }).collect()
    }
    
    /// Ice cake conflict: last writer wins with quorum
    pub fn resolve_ice_cake_conflict(
        &self,
        local: &IceCakedDecision,
        remote: &IceCakedDecision,
    ) -> ConflictResolution {
        if remote.ice_caked_at_cycle > local.ice_caked_at_cycle {
            ConflictResolution::AcceptRemote
        } else if local.ice_caked_at_cycle > remote.ice_caked_at_cycle {
            ConflictResolution::KeepLocal
        } else {
            // Same cycle: compare gate states
            match (&local.gate_state, &remote.gate_state) {
                (GateState::Flow, GateState::Flow) => ConflictResolution::KeepLocal,
                (GateState::Block, _) => ConflictResolution::KeepLocal,
                (_, GateState::Block) => ConflictResolution::AcceptRemote,
                _ => ConflictResolution::Conflict("Simultaneous ice cake".into()),
            }
        }
    }
}
```

Right now each ladybug instance is isolated. For AGI that accumulates across *many* agents, need federation:
- CRDT for moment merge (eventual consistency)
- Ice cake conflict resolution (last-writer-wins? quorum?)
- Cross-instance resonance search

**Location**: `src/federation/`

**Dependencies**: `crdts`, `libp2p` (optional), `tokio`

---

### 6. Causal Graph — Beyond SPO

You have NARS for inference, but the *causal structure* is implicit. Make it explicit:

```rust
// MISSING: Structural Causal Model
pub struct CausalGraph {
    nodes: HashMap<String, CausalNode>,
    edges: HashMap<(String, String), CausalEdge>,
}

pub struct CausalNode {
    pub id: String,
    pub fingerprint: Fingerprint,
    pub observed_value: Option<Fingerprint>,
    pub is_intervention: bool,
}

pub struct CausalEdge {
    pub strength: TruthValue,      // NARS truth
    pub mechanism: Fingerprint,    // HOW the causation works
    pub interventional: bool,      // Observed or intervened?
}

impl CausalGraph {
    /// Do-calculus intervention: P(Y | do(X=x))
    pub fn do_intervention(&mut self, node: &str, value: Fingerprint) -> &mut Self {
        // Remove incoming edges (breaks observational correlation)
        self.edges.retain(|(_, to), _| to != node);
        
        // Set value
        if let Some(n) = self.nodes.get_mut(node) {
            n.observed_value = Some(value);
            n.is_intervention = true;
        }
        
        self
    }
    
    /// Counterfactual query: "What would Y be if X had been x?"
    pub fn counterfactual(&self, target: &str, intervention: &str, value: Fingerprint) -> TruthValue {
        // 1. Abduction: infer exogenous variables from observations
        let exogenous = self.abduct_exogenous();
        
        // 2. Action: apply intervention
        let mut modified = self.clone();
        modified.do_intervention(intervention, value);
        
        // 3. Prediction: propagate forward
        modified.propagate_with(&exogenous);
        
        // Return truth value of target
        modified.nodes.get(target)
            .map(|n| n.compute_truth())
            .unwrap_or(TruthValue::unknown())
    }
    
    /// Forward propagation through causal graph
    fn propagate(&mut self) {
        // Topological sort
        let order = self.topological_sort();
        
        for node_id in order {
            let parents: Vec<_> = self.edges.iter()
                .filter(|((_, to), _)| to == &node_id)
                .map(|((from, _), edge)| (from.clone(), edge.clone()))
                .collect();
            
            if !parents.is_empty() {
                let value = self.compute_from_parents(&node_id, &parents);
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.observed_value = Some(value);
                }
            }
        }
    }
}
```

You have `world.fork()` but that's branching state, not causal inference. Real counterfactuals need do-calculus.

**Location**: `src/causal/`

**Dependencies**: `petgraph` (for DAG operations)

---

### 7. Extension Bridge — Make Extensions Talk

```
Current:           What's needed:
                   
codebook           codebook
   ↓                  ↕
(isolated)      ┌─────┴─────┐
               spo ←───────→ hologram
hologram          ↕           ↕
   ↓           compress ←───→ langextract
(isolated)            ↘   ↙
                     core
```

Extensions are silos. Need unified stack:

```rust
// MISSING: Extension bridge
pub struct CognitiveStack {
    core: Database,
    codebook: Option<Codebook>,
    spo: Option<SPOGrid>,
    hologram: Option<CrystalMemory>,
    compress: Option<Compressor>,
    langextract: Option<GrammarTriangle>,
}

impl CognitiveStack {
    /// Full pipeline: input → grammar → SPO → fingerprint → codebook lookup → hologram store
    pub fn ingest(&mut self, input: Input) -> Result<Moment> {
        // 1. Parse to grammar (if langextract available)
        let triples = if let Some(ref mut le) = self.langextract {
            le.parse(&input)?
        } else {
            vec![input.to_default_triple()]
        };
        
        // 2. Store in SPO (if available)
        if let Some(ref mut spo) = self.spo {
            for triple in &triples {
                spo.store(&triple.subject, &triple.predicate, &triple.object, &triple.qualia)?;
            }
        }
        
        // 3. Generate fingerprint
        let fp = triples.to_fingerprint();
        
        // 4. Codebook lookup (if available)
        let slot = self.codebook.as_ref().map(|cb| cb.lookup(&fp));
        
        // 5. Hologram store (if available)
        if let Some(ref mut holo) = self.hologram {
            holo.store(&fp)?;
        }
        
        // 6. Core storage
        let moment = Moment::new(&fp);
        self.core.store_moment(&moment)?;
        
        Ok(moment)
    }
    
    /// Unified query across all extensions
    pub fn query(&self, q: UnifiedQuery) -> Result<Vec<UnifiedResult>> {
        match q {
            UnifiedQuery::Sql(sql) => self.core.sql(&sql),
            UnifiedQuery::Cypher(cypher) => self.core.cypher(&cypher),
            UnifiedQuery::Resonate(fp, threshold) => self.core.resonate(&fp, threshold),
            UnifiedQuery::Spo(s, p, o) => self.spo.as_ref().map(|spo| spo.query(s, p, o)),
            UnifiedQuery::Codebook(fp) => self.codebook.as_ref().map(|cb| cb.lookup(&fp)),
        }
    }
}
```

**Location**: `src/stack.rs`

**Dependencies**: All extension features

---

### 8. Self-Modeling — The Missing Meta-Layer

```rust
// MISSING: Self-model
pub struct SelfModel {
    /// What I know about my own processing
    pub processing_style: ThinkingStyle,
    
    /// What resonance patterns work for me
    pub effective_threshold: f32,
    
    /// My sweet spot parameters (learned from experience)
    pub sweet_spot_center: f32,
    pub sweet_spot_width: f32,
    
    /// My ice-caked commitments
    pub core_beliefs: Vec<IceCakedDecision>,
    
    /// What I'm uncertain about
    pub open_questions: Vec<Fingerprint>,
    
    /// Statistics about my learning
    pub total_moments: u64,
    pub total_breakthroughs: u64,
    pub average_struggle_duration: f32,
    pub domains_explored: Vec<String>,
}

impl SelfModel {
    /// Update from completed session
    pub fn update_from_session(&mut self, session: &LearningSession) {
        self.total_moments += session.moments.len() as u64;
        self.total_breakthroughs += session.breakthroughs().len() as u64;
        
        // Learn effective threshold from successful resonances
        let successful_resonances: Vec<f32> = session.moments.iter()
            .filter(|m| m.is_breakthrough())
            .filter_map(|m| m.resonance_score)
            .collect();
        
        if !successful_resonances.is_empty() {
            let mean: f32 = successful_resonances.iter().sum::<f32>() / successful_resonances.len() as f32;
            // Exponential moving average
            self.effective_threshold = 0.9 * self.effective_threshold + 0.1 * mean;
        }
        
        // Track struggle patterns
        let struggles: Vec<&Moment> = session.moments.iter()
            .filter(|m| m.qualia.is_struggle())
            .collect();
        
        if !struggles.is_empty() {
            // Could compute average duration, common domains, etc.
        }
    }
    
    /// Generate self-report
    pub fn introspect(&self) -> String {
        format!(
            r#"# Self-Model Report

## Processing Style
Current: {:?}
Effective resonance threshold: {:.2}
Sweet spot: center={:.1}s, width={:.1}s

## Experience
Total moments: {}
Breakthroughs: {} ({:.1}% rate)
Average struggle duration: {:.1}s
Domains: {}

## Core Beliefs (Ice-Caked)
{}

## Open Questions
{}
"#,
            self.processing_style,
            self.effective_threshold,
            self.sweet_spot_center,
            self.sweet_spot_width,
            self.total_moments,
            self.total_breakthroughs,
            self.total_breakthroughs as f32 / self.total_moments.max(1) as f32 * 100.0,
            self.average_struggle_duration,
            self.domains_explored.join(", "),
            self.core_beliefs.iter()
                .map(|b| format!("- {} ({})", b.content, b.gate_state))
                .collect::<Vec<_>>()
                .join("\n"),
            self.open_questions.iter()
                .take(5)
                .map(|q| format!("- {:?}...", &q.as_raw()[..4]))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
    
    /// Am I good at this domain?
    pub fn confidence_in_domain(&self, domain: &str) -> f32 {
        if self.domains_explored.contains(&domain.to_string()) {
            // Could track per-domain breakthrough rates
            0.7
        } else {
            0.3
        }
    }
}
```

The system captures moments. But does it *model itself*? Can it say "I tend to struggle with X" or "My resonance threshold is too high"?

**Location**: `src/self_model.rs`

**Dependencies**: None (pure Rust)

---

## Unified Entry Point

```rust
/// The universal cognitive substrate
pub struct Ladybug {
    /// Core database
    pub db: Database,
    
    /// Extension stack
    pub stack: CognitiveStack,
    
    /// Self-model
    pub self_model: SelfModel,
    
    /// Federation (optional)
    pub federation: Option<FederationClient>,
    
    /// Causal graph (optional)
    pub causal: Option<CausalGraph>,
    
    /// Temporal engine
    pub temporal: TemporalResonanceEngine,
}

impl Ladybug {
    /// Open or create database
    pub async fn open(path: &str) -> Result<Self> {
        let db = Database::open(path).await?;
        Ok(Self {
            db,
            stack: CognitiveStack::default(),
            self_model: SelfModel::default(),
            federation: None,
            causal: None,
            temporal: TemporalResonanceEngine::default(),
        })
    }
    
    /// True universal: any input, any modality
    pub async fn ingest(&mut self, input: impl Into<Input>) -> Result<Moment> {
        self.stack.ingest(input.into())
    }
    
    /// Resonance with temporal + federated search
    pub async fn resonate_universal(
        &self,
        query: &Fingerprint,
        threshold: f32,
        limit: usize,
    ) -> Vec<UniversalMatch> {
        let mut results = Vec::new();
        
        // Local temporal resonance
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let temporal = self.temporal.find_temporally_relevant(query, now, threshold, limit);
        results.extend(temporal.into_iter().map(|(i, score)| UniversalMatch {
            source: MatchSource::Local,
            index: i,
            score,
        }));
        
        // Federated resonance (if available)
        if let Some(ref fed) = self.federation {
            let federated = fed.federated_resonate(query, threshold).await;
            results.extend(federated.into_iter().map(|m| UniversalMatch {
                source: MatchSource::Federated(m.peer_id),
                index: 0,  // Would need proper indexing
                score: m.resonance,
            }));
        }
        
        // Sort by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }
    
    /// Causal counterfactual
    pub fn what_if(&self, intervention: &str, value: Fingerprint) -> Option<TruthValue> {
        self.causal.as_ref().map(|c| {
            let parts: Vec<_> = intervention.split('=').collect();
            if parts.len() == 2 {
                c.counterfactual(parts[1].trim(), parts[0].trim(), value)
            } else {
                TruthValue::unknown()
            }
        })
    }
    
    /// Self-report
    pub fn who_am_i(&self) -> &SelfModel {
        &self.self_model
    }
    
    /// Update self-model after session
    pub fn learn_about_self(&mut self, session: &LearningSession) {
        self.self_model.update_from_session(session);
    }
}

pub struct UniversalMatch {
    pub source: MatchSource,
    pub index: usize,
    pub score: f32,
}

pub enum MatchSource {
    Local,
    Federated(String),
}
```

---

## Feature Flags

```toml
# Cargo.toml
[features]
default = ["core"]

# Core (always included)
core = []

# Grammar parsing (universal input)
grammar = ["tree-sitter", "tree-sitter-rust", "tree-sitter-python", "tree-sitter-javascript"]

# Local modality inference (no API calls)
modality = ["candle", "candle-transformers", "whisper-rs"]

# Temporal resonance
temporal = []

# Multi-instance federation
federation = ["crdts", "libp2p", "tokio"]

# Structural causal models
causal = ["petgraph"]

# Self-modeling / introspection
self-model = []

# All extensions
extensions = ["codebook", "hologram", "spo", "compress"]

# Everything
full = ["grammar", "modality", "temporal", "federation", "causal", "self-model", "extensions"]
```

---

## Summary

| Layer | Status | Priority | Effort |
|-------|--------|----------|--------|
| Grammar parsing | ❌ Missing | HIGH | Medium |
| Modality adapters | ❌ Missing | HIGH | High |
| Projection operators | ❌ Missing | MEDIUM | Low |
| Temporal resonance | 🟡 Partial | MEDIUM | Low |
| Federation | ❌ Missing | LOW | High |
| Causal inference | ❌ Missing | MEDIUM | Medium |
| Extension bridge | ❌ Missing | HIGH | Medium |
| Self-modeling | ❌ Missing | LOW | Low |

**The gap**: You have the *substrate*. You need the *surface* that touches everything.

**One-line summary**: Unified cognitive database exists. Universal input layer doesn't.

---

## Next Steps

1. **Grammar layer first** — Without parsing, everything needs external APIs
2. **Extension bridge** — Make existing extensions talk to each other
3. **Projection operators** — Accept any embedding space
4. **Temporal resonance** — Time-weighted sweet spot
5. **Self-model** — Basic introspection
6. **Federation** — Later, when single-instance is complete
7. **Causal** — Later, research-grade complexity

---

*Last updated: 2026-01-30*
