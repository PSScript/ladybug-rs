//! Search module - Alien Magic Vector Search + Causal Reasoning
//!
//! This module provides a search API that looks like float vector search
//! (Faiss, Annoy, etc.) but runs on pure integer SIMD operations.
//!
//! # Features
//!
//! - **HDR Cascade**: Hierarchical filtering (1-bit → 4-bit → 8-bit → exact)
//! - **Mexican Hat**: Discrimination with excitation and inhibition zones
//! - **Rolling σ**: Window-based coherence detection
//! - **A⊗B⊗B=A**: O(1) direct retrieval via XOR unbinding
//! - **Causal Search**: Three rungs (correlate, intervene, counterfact)
//!
//! # Example
//!
//! ```ignore
//! use ladybug::search::{AlienSearch, CausalSearch, QueryMode};
//!
//! // Correlation search (looks like vector search)
//! let mut search = AlienSearch::with_capacity(10000);
//! let results = search.search_similarity(&query.to_words(), 10);
//!
//! // Causal search (three rungs)
//! let mut causal = CausalSearch::new();
//!
//! // Rung 1: What correlates?
//! let correlates = causal.query_correlates(&x, 10);
//!
//! // Rung 2: What happens if I do this?
//! causal.store_intervention(&state, &action, &outcome, 1.0);
//! let outcomes = causal.query_outcome(&state, &action);
//!
//! // Rung 3: What would have happened?
//! let counterfactuals = causal.query_counterfactual(&state, &alt_action);
//! ```

pub mod hdr_cascade;
pub mod causal;

pub use hdr_cascade::{
    // Core operations
    hamming_distance,
    sketch_1bit,
    sketch_4bit,
    sketch_8bit,
    sketch_1bit_sum,
    sketch_4bit_sum,
    sketch_8bit_sum,
    
    // Mexican hat
    MexicanHat,
    
    // Rolling window
    RollingWindow,
    
    // HDR index
    HdrIndex,
    
    // Bound retrieval (A⊗B⊗B=A)
    BoundRetrieval,
    
    // Unified API (the alien magic)
    SearchResult,
    AlienSearch,
    
    // Fingerprint extension trait
    FingerprintSearch,
};

pub use causal::{
    // Verbs
    CausalVerbs,
    
    // Edge types
    EdgeType,
    CausalEdge,
    
    // Rung stores
    CorrelationStore,
    InterventionStore,
    CounterfactualStore,
    
    // Unified API
    QueryMode,
    CausalResult,
    CausalSearch,
};
