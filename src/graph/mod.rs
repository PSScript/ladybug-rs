//! Graph module - Cognitive Graph Substrate
//!
//! Nodes = Fingerprints (concepts, entities, states)
//! Edges = Fingerprint ⊗ Verb ⊗ Fingerprint (relationships)
//! Verbs = 144 core relations on Go board topology

mod edge;
mod traversal;
pub mod cognitive;

pub use edge::{Edge, EdgeType};
pub use traversal::Traversal;
pub use cognitive::{
    Verb, VerbCategory, CogNode, NodeType, CogEdge, CogGraph,
};

pub mod avx_engine;
pub use avx_engine::{
    FingerprintGraph,
    QueryMatch,
    hamming_distance,
    batched_query,
    avx512_available,
    simd_info,
};
