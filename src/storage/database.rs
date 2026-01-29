//! Main Database API - unified interface for all operations

use crate::core::{Fingerprint, HammingEngine};
use crate::cognitive::Thought;
use crate::nars::TruthValue;
use crate::graph::{Edge, Traversal};
use crate::query::{Query, QueryResult};
use crate::{Result, Error};

use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;

/// Main database handle - unified access to all operations
pub struct Database {
    /// Path to database
    path: String,
    /// Hamming search engine (pre-indexed)
    hamming: Arc<RwLock<HammingEngine>>,
    /// Current version (for copy-on-write)
    version: u64,
}

impl Database {
    /// Open or create a database
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        
        // Create directory if needed
        std::fs::create_dir_all(&path_str)?;
        
        Ok(Self {
            path: path_str,
            hamming: Arc::new(RwLock::new(HammingEngine::new())),
            version: 0,
        })
    }
    
    /// Connect to in-memory database
    pub fn memory() -> Self {
        Self {
            path: ":memory:".to_string(),
            hamming: Arc::new(RwLock::new(HammingEngine::new())),
            version: 0,
        }
    }
    
    // === Conventional Operations ===
    
    /// Execute SQL query
    pub fn sql(&self, query: &str) -> Result<QueryResult> {
        // TODO: Integrate with DataFusion
        let _ = query;
        Ok(QueryResult {
            rows: vec![],
            columns: vec![],
        })
    }
    
    /// Execute Cypher query (transpiled to SQL)
    pub fn cypher(&self, query: &str) -> Result<QueryResult> {
        // TODO: Cypher parser + transpiler
        let _ = query;
        Ok(QueryResult {
            rows: vec![],
            columns: vec![],
        })
    }
    
    /// Vector similarity search (ANN)
    pub fn vector_search(&self, _embedding: &[f32], _k: usize) -> Result<Vec<String>> {
        // TODO: Lance vector index
        Ok(vec![])
    }
    
    // === AGI Operations ===
    
    /// Resonance search (Hamming similarity)
    pub fn resonate(
        &self,
        fingerprint: &Fingerprint,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let engine = self.hamming.read();
        engine.search_threshold(fingerprint, threshold, limit)
            .into_iter()
            .map(|(idx, _, sim)| (idx, sim))
            .collect()
    }
    
    /// Resonate by content (auto-generates fingerprint)
    pub fn resonate_content(
        &self,
        content: &str,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, f32)> {
        let fp = Fingerprint::from_content(content);
        self.resonate(&fp, threshold, limit)
    }
    
    /// Index fingerprints for resonance search
    pub fn index_fingerprints(&self, fingerprints: Vec<Fingerprint>) {
        let mut engine = self.hamming.write();
        engine.index(fingerprints);
    }
    
    /// Start a graph traversal query
    pub fn traverse(&self, start_id: &str) -> Traversal {
        Traversal::from(start_id)
    }
    
    /// Fork database for counterfactual reasoning
    pub fn fork(&self) -> Database {
        Database {
            path: self.path.clone(),
            hamming: Arc::clone(&self.hamming),
            version: self.version + 1,
        }
    }
    
    /// Detect butterfly effects (causal amplification chains)
    pub fn detect_butterflies(
        &self,
        source_id: &str,
        threshold: f32,
        max_depth: usize,
    ) -> Result<Vec<(Vec<String>, f32)>> {
        // TODO: Recursive CTE query for amplification chains
        let _ = (source_id, threshold, max_depth);
        Ok(vec![])
    }
    
    // === CRUD Operations ===
    
    /// Add a thought
    pub fn add_thought(&self, thought: &Thought) -> Result<String> {
        // TODO: Lance insert
        Ok(thought.id.clone())
    }
    
    /// Add an edge
    pub fn add_edge(&self, edge: &Edge) -> Result<()> {
        // TODO: Lance insert
        let _ = edge;
        Ok(())
    }
    
    /// Get thought by ID
    pub fn get_thought(&self, id: &str) -> Result<Option<Thought>> {
        // TODO: Lance lookup
        let _ = id;
        Ok(None)
    }
    
    // === Database Info ===
    
    /// Database path
    pub fn path(&self) -> &str {
        &self.path
    }
    
    /// Current version
    pub fn version(&self) -> u64 {
        self.version
    }
    
    /// Number of indexed fingerprints
    pub fn fingerprint_count(&self) -> usize {
        self.hamming.read().len()
    }
}

// Convenience function
pub fn open<P: AsRef<Path>>(path: P) -> Result<Database> {
    Database::open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_open_memory() {
        let db = Database::memory();
        assert_eq!(db.path(), ":memory:");
    }
    
    #[test]
    fn test_resonate() {
        let db = Database::memory();
        
        // Index some fingerprints
        let fps: Vec<Fingerprint> = (0..100)
            .map(|i| Fingerprint::from_content(&format!("thought_{}", i)))
            .collect();
        db.index_fingerprints(fps);
        
        // Search
        let query = Fingerprint::from_content("thought_50");
        let results = db.resonate(&query, 0.5, 10);
        
        // Should find exact match with similarity 1.0
        assert!(!results.is_empty());
        assert!(results[0].1 > 0.99);
    }
    
    #[test]
    fn test_fork() {
        let db = Database::memory();
        let forked = db.fork();
        
        assert_eq!(forked.version(), db.version() + 1);
    }
}
