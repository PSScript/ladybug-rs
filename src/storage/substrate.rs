//! Substrate - Unified Interface DTO
//!
//! The Substrate bridges all storage layers into a single coherent interface.
//! Lance is the source of truth, BindSpace is the hot cache.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           SUBSTRATE                                         │
//! │                    (Unified Interface DTO)                                  │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │                                                                             │
//! │   Query Adapters                                                            │
//! │   ├── RedisAdapter    (Redis syntax → Substrate ops)                       │
//! │   ├── SqlAdapter      (SQL → DataFusion → Substrate)                       │
//! │   └── CypherAdapter   (Cypher → Graph ops → Substrate)                     │
//! │                                                                             │
//! │   Hot Layer (BindSpace)                                                     │
//! │   ├── Surface (0x00-0x0F): 4,096 CAM operations                            │
//! │   ├── Fluid (0x10-0x7F): 28,672 working memory                             │
//! │   └── Nodes (0x80-0xFF): 32,768 hot node cache                             │
//! │                                                                             │
//! │   Cold Layer (LanceDB)                                                      │
//! │   ├── nodes.lance     → All nodes with fingerprints                        │
//! │   ├── edges.lance     → All edges with weights                             │
//! │   └── sessions.lance  → Consciousness snapshots                            │
//! │                                                                             │
//! │   Sync Layer                                                                │
//! │   ├── Write-through to Lance on commit                                     │
//! │   ├── Read from BindSpace if hot, else Lance                               │
//! │   └── Promote hot reads, evict cold                                        │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let substrate = Substrate::new(config).await?;
//!
//! // Write (goes to both BindSpace and queued for Lance)
//! let addr = substrate.write(fingerprint, label).await?;
//!
//! // Read (checks BindSpace first, falls back to Lance)
//! let node = substrate.read(addr).await?;
//!
//! // Search (uses HDR cascade on BindSpace, falls back to Lance ANN)
//! let results = substrate.search(query, k).await?;
//!
//! // Commit (flushes pending writes to Lance)
//! substrate.commit().await?;
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

use super::bind_space::{BindSpace, BindNode, BindEdge, Addr, FINGERPRINT_WORDS, hamming_distance};
use super::cog_redis::{CogAddr, CogValue, CogEdge, Tier};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Substrate configuration
#[derive(Debug, Clone)]
pub struct SubstrateConfig {
    /// Path to Lance database
    pub lance_path: PathBuf,
    /// Maximum hot nodes in BindSpace
    pub max_hot_nodes: usize,
    /// TTL for fluid zone entries
    pub fluid_ttl: Duration,
    /// Promotion threshold (access count before promoting to node)
    pub promotion_threshold: u32,
    /// Demotion threshold (time since last access before demoting)
    pub demotion_threshold: Duration,
    /// Write buffer size before auto-commit
    pub write_buffer_size: usize,
    /// Enable async commit
    pub async_commit: bool,
    /// Search cascade levels
    pub cascade_levels: usize,
}

impl Default for SubstrateConfig {
    fn default() -> Self {
        Self {
            lance_path: PathBuf::from("./data/lance"),
            max_hot_nodes: 32768,  // Full node space
            fluid_ttl: Duration::from_secs(300),  // 5 minutes
            promotion_threshold: 10,
            demotion_threshold: Duration::from_secs(3600),  // 1 hour
            write_buffer_size: 1000,
            async_commit: true,
            cascade_levels: 4,
        }
    }
}

// =============================================================================
// SUBSTRATE NODE (Unified representation)
// =============================================================================

/// A node in the substrate (unified view)
#[derive(Clone, Debug)]
pub struct SubstrateNode {
    /// 16-bit cognitive address
    pub addr: CogAddr,
    /// 10K-bit fingerprint (156 × 64-bit words)
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Optional label
    pub label: Option<String>,
    /// Qualia index (0-255)
    pub qidx: u8,
    /// Access count
    pub access_count: u32,
    /// Last access time
    pub last_access: Instant,
    /// Creation time
    pub created: Instant,
    /// TTL (None = permanent in node space)
    pub ttl: Option<Duration>,
    /// Lance row ID (if persisted)
    pub lance_id: Option<String>,
    /// Version for MVCC
    pub version: u64,
    /// Is this node dirty (needs Lance sync)?
    pub dirty: bool,
}

impl SubstrateNode {
    /// Create a new node with fingerprint
    pub fn new(addr: CogAddr, fingerprint: [u64; FINGERPRINT_WORDS]) -> Self {
        Self {
            addr,
            fingerprint,
            label: None,
            qidx: 128,  // Neutral qualia
            access_count: 0,
            last_access: Instant::now(),
            created: Instant::now(),
            ttl: None,
            lance_id: None,
            version: 1,
            dirty: true,  // New nodes are dirty
        }
    }

    /// Create with label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set qualia index
    pub fn with_qidx(mut self, qidx: u8) -> Self {
        self.qidx = qidx;
        self
    }

    /// Set TTL for fluid zone
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Record an access
    pub fn touch(&mut self) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_access = Instant::now();
    }

    /// Check if expired (for fluid zone)
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.last_access.elapsed() > ttl
        } else {
            false
        }
    }

    /// Should promote from fluid to node?
    pub fn should_promote(&self, threshold: u32) -> bool {
        self.ttl.is_some() && self.access_count >= threshold
    }

    /// Should demote from node to fluid?
    pub fn should_demote(&self, threshold: Duration) -> bool {
        self.ttl.is_none() && self.last_access.elapsed() > threshold
    }

    /// Popcount of fingerprint
    pub fn popcount(&self) -> u32 {
        self.fingerprint.iter().map(|w| w.count_ones()).sum()
    }

    /// Hamming distance to another node
    pub fn distance_to(&self, other: &SubstrateNode) -> u32 {
        hamming_distance(&self.fingerprint, &other.fingerprint)
    }

    /// Similarity to another node (0.0-1.0)
    pub fn similarity_to(&self, other: &SubstrateNode) -> f32 {
        let dist = self.distance_to(other);
        1.0 - (dist as f32 / 10000.0)
    }
}

impl From<&BindNode> for SubstrateNode {
    fn from(node: &BindNode) -> Self {
        Self {
            addr: CogAddr::new(0),  // Will be set by caller
            fingerprint: node.fingerprint,
            label: node.label.clone(),
            qidx: node.qidx,
            access_count: node.access_count,
            last_access: Instant::now(),
            created: Instant::now(),
            ttl: None,
            lance_id: None,
            version: 1,
            dirty: false,
        }
    }
}

// =============================================================================
// SUBSTRATE EDGE (Unified edge representation)
// =============================================================================

/// An edge in the substrate
#[derive(Clone, Debug)]
pub struct SubstrateEdge {
    /// Source node address
    pub from: CogAddr,
    /// Target node address
    pub to: CogAddr,
    /// Verb/relation address (in surface tier)
    pub verb: CogAddr,
    /// Bound fingerprint: from ⊗ verb ⊗ to
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Edge weight
    pub weight: f32,
    /// Creation time
    pub created: Instant,
    /// Lance ID (if persisted)
    pub lance_id: Option<String>,
    /// Is dirty?
    pub dirty: bool,
}

impl SubstrateEdge {
    /// Create a new edge
    pub fn new(from: CogAddr, verb: CogAddr, to: CogAddr) -> Self {
        Self {
            from,
            to,
            verb,
            fingerprint: [0u64; FINGERPRINT_WORDS],
            weight: 1.0,
            created: Instant::now(),
            lance_id: None,
            dirty: true,
        }
    }

    /// Bind fingerprints
    pub fn bind(
        &mut self,
        from_fp: &[u64; FINGERPRINT_WORDS],
        verb_fp: &[u64; FINGERPRINT_WORDS],
        to_fp: &[u64; FINGERPRINT_WORDS],
    ) {
        for i in 0..FINGERPRINT_WORDS {
            self.fingerprint[i] = from_fp[i] ^ verb_fp[i] ^ to_fp[i];
        }
    }

    /// Unbind to recover unknown from known
    pub fn unbind(
        &self,
        known: &[u64; FINGERPRINT_WORDS],
        verb_fp: &[u64; FINGERPRINT_WORDS],
    ) -> [u64; FINGERPRINT_WORDS] {
        let mut result = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            result[i] = self.fingerprint[i] ^ known[i] ^ verb_fp[i];
        }
        result
    }
}

// =============================================================================
// WRITE BUFFER (Pending writes to Lance)
// =============================================================================

/// A pending write operation
#[derive(Clone, Debug)]
pub enum WriteOp {
    /// Insert/update a node
    UpsertNode(SubstrateNode),
    /// Delete a node
    DeleteNode(CogAddr),
    /// Insert/update an edge
    UpsertEdge(SubstrateEdge),
    /// Delete an edge
    DeleteEdge { from: CogAddr, verb: CogAddr, to: CogAddr },
}

/// Write buffer for batching Lance operations
#[derive(Debug)]
pub struct WriteBuffer {
    /// Pending operations
    ops: Vec<WriteOp>,
    /// Maximum size before auto-flush
    max_size: usize,
    /// Total bytes queued
    bytes_queued: usize,
}

impl WriteBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            ops: Vec::with_capacity(max_size),
            max_size,
            bytes_queued: 0,
        }
    }

    /// Add an operation
    pub fn push(&mut self, op: WriteOp) -> bool {
        self.ops.push(op);
        self.ops.len() >= self.max_size
    }

    /// Take all operations
    pub fn drain(&mut self) -> Vec<WriteOp> {
        std::mem::take(&mut self.ops)
    }

    /// Number of pending operations
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Is buffer empty?
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

// =============================================================================
// SUBSTRATE STATISTICS
// =============================================================================

/// Substrate statistics
#[derive(Debug, Default)]
pub struct SubstrateStats {
    /// Hot cache hits
    pub hot_hits: AtomicU64,
    /// Hot cache misses (fell through to Lance)
    pub hot_misses: AtomicU64,
    /// Nodes in hot cache
    pub hot_nodes: AtomicU64,
    /// Edges in hot cache
    pub hot_edges: AtomicU64,
    /// Pending writes
    pub pending_writes: AtomicU64,
    /// Lance reads
    pub lance_reads: AtomicU64,
    /// Lance writes
    pub lance_writes: AtomicU64,
    /// Promotions (fluid → node)
    pub promotions: AtomicU64,
    /// Demotions (node → fluid)
    pub demotions: AtomicU64,
    /// Evictions (expired from fluid)
    pub evictions: AtomicU64,
}

impl SubstrateStats {
    /// Hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hot_hits.load(Ordering::Relaxed) as f64;
        let misses = self.hot_misses.load(Ordering::Relaxed) as f64;
        if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        }
    }
}

// =============================================================================
// SUBSTRATE (The unified interface)
// =============================================================================

/// The Substrate - unified interface to all storage layers
pub struct Substrate {
    /// Configuration
    config: SubstrateConfig,

    /// Hot cache (BindSpace)
    bind_space: RwLock<BindSpace>,

    /// Extended hot cache for nodes beyond 64K
    extended_nodes: RwLock<HashMap<u64, SubstrateNode>>,

    /// Extended edges
    extended_edges: RwLock<Vec<SubstrateEdge>>,

    /// Write buffer (pending Lance writes)
    write_buffer: Mutex<WriteBuffer>,

    /// Next extended node ID
    next_extended_id: AtomicU64,

    /// Version counter
    version: AtomicU64,

    /// Statistics
    stats: SubstrateStats,

    /// Is Lance connected?
    lance_connected: AtomicBool,
}

impl Substrate {
    /// Create a new substrate
    pub fn new(config: SubstrateConfig) -> Self {
        let write_buffer = WriteBuffer::new(config.write_buffer_size);

        Self {
            config,
            bind_space: RwLock::new(BindSpace::new()),
            extended_nodes: RwLock::new(HashMap::new()),
            extended_edges: RwLock::new(Vec::new()),
            write_buffer: Mutex::new(write_buffer),
            next_extended_id: AtomicU64::new(0x10000),  // Start after 16-bit space
            version: AtomicU64::new(1),
            stats: SubstrateStats::default(),
            lance_connected: AtomicBool::new(false),
        }
    }

    /// Create with default config
    pub fn default_new() -> Self {
        Self::new(SubstrateConfig::default())
    }

    // =========================================================================
    // READ OPERATIONS
    // =========================================================================

    /// Read a node by address
    pub fn read(&self, addr: CogAddr) -> Option<SubstrateNode> {
        // Check hot cache first
        let bind_space = self.bind_space.read().unwrap();
        let bind_addr = Addr::from(addr.0);

        if let Some(node) = bind_space.read(bind_addr) {
            self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
            let mut result = SubstrateNode::from(node);
            result.addr = addr;
            return Some(result);
        }

        // Check extended nodes
        drop(bind_space);
        let extended = self.extended_nodes.read().unwrap();
        if let Some(node) = extended.get(&(addr.0 as u64)) {
            self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
            return Some(node.clone());
        }

        // Cache miss - would query Lance here
        self.stats.hot_misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Read a node by label
    pub fn read_by_label(&self, label: &str) -> Option<SubstrateNode> {
        // Scan hot cache
        let bind_space = self.bind_space.read().unwrap();

        // Check surfaces
        for prefix in 0..16u8 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    if node.label.as_deref() == Some(label) {
                        self.stats.hot_hits.fetch_add(1, Ordering::Relaxed);
                        let mut result = SubstrateNode::from(node);
                        result.addr = CogAddr::from_parts(prefix, slot);
                        return Some(result);
                    }
                }
            }
        }

        // Scan nodes (would be more efficient with index)
        // For now, just return None - in production this would query Lance
        None
    }

    // =========================================================================
    // WRITE OPERATIONS
    // =========================================================================

    /// Write a node, returns its address
    pub fn write(&self, fingerprint: [u64; FINGERPRINT_WORDS]) -> CogAddr {
        let mut bind_space = self.bind_space.write().unwrap();
        let addr = bind_space.write(fingerprint);

        let node = SubstrateNode::new(CogAddr::from(addr.0), fingerprint);

        // Queue for Lance
        let mut buffer = self.write_buffer.lock().unwrap();
        let should_flush = buffer.push(WriteOp::UpsertNode(node));
        drop(buffer);

        self.stats.hot_nodes.fetch_add(1, Ordering::Relaxed);
        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

        if should_flush {
            drop(bind_space);
            self.flush_sync();
        }

        CogAddr::from(addr.0)
    }

    /// Write a node with label
    pub fn write_labeled(&self, fingerprint: [u64; FINGERPRINT_WORDS], label: &str) -> CogAddr {
        let mut bind_space = self.bind_space.write().unwrap();
        let addr = bind_space.write_labeled(fingerprint, label);

        let node = SubstrateNode::new(CogAddr::from(addr.0), fingerprint)
            .with_label(label);

        // Queue for Lance
        let mut buffer = self.write_buffer.lock().unwrap();
        buffer.push(WriteOp::UpsertNode(node));

        self.stats.hot_nodes.fetch_add(1, Ordering::Relaxed);
        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

        CogAddr::from(addr.0)
    }

    /// Write to fluid zone with TTL
    pub fn write_fluid(&self, fingerprint: [u64; FINGERPRINT_WORDS], ttl: Duration) -> CogAddr {
        let addr = self.write(fingerprint);

        // The address will be in fluid zone automatically if we track TTL
        // For now, store in extended with TTL
        let mut node = SubstrateNode::new(addr, fingerprint);
        node.ttl = Some(ttl);

        let mut extended = self.extended_nodes.write().unwrap();
        extended.insert(addr.0 as u64, node);

        addr
    }

    /// Delete a node
    pub fn delete(&self, addr: CogAddr) -> bool {
        let mut bind_space = self.bind_space.write().unwrap();
        let bind_addr = Addr::from(addr.0);

        let deleted = bind_space.delete(bind_addr).is_some();

        if deleted {
            // Queue delete for Lance
            let mut buffer = self.write_buffer.lock().unwrap();
            buffer.push(WriteOp::DeleteNode(addr));
            self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);
        }

        // Also check extended
        drop(bind_space);
        let mut extended = self.extended_nodes.write().unwrap();
        extended.remove(&(addr.0 as u64));

        deleted
    }

    // =========================================================================
    // EDGE OPERATIONS
    // =========================================================================

    /// Create an edge
    pub fn link(&self, from: CogAddr, verb: CogAddr, to: CogAddr) -> Option<SubstrateEdge> {
        let mut bind_space = self.bind_space.write().unwrap();

        let from_addr = Addr::from(from.0);
        let verb_addr = Addr::from(verb.0);
        let to_addr = Addr::from(to.0);

        // Get fingerprints
        let from_fp = bind_space.read(from_addr).map(|n| n.fingerprint)?;
        let verb_fp = bind_space.read(verb_addr).map(|n| n.fingerprint)?;
        let to_fp = bind_space.read(to_addr).map(|n| n.fingerprint)?;

        // Create edge in BindSpace
        bind_space.link(from_addr, verb_addr, to_addr);

        // Create substrate edge
        let mut edge = SubstrateEdge::new(from, verb, to);
        edge.bind(&from_fp, &verb_fp, &to_fp);

        // Queue for Lance
        let mut buffer = self.write_buffer.lock().unwrap();
        buffer.push(WriteOp::UpsertEdge(edge.clone()));

        self.stats.hot_edges.fetch_add(1, Ordering::Relaxed);
        self.stats.pending_writes.fetch_add(1, Ordering::Relaxed);

        Some(edge)
    }

    /// Get outgoing edges
    pub fn edges_out(&self, from: CogAddr) -> Vec<SubstrateEdge> {
        let bind_space = self.bind_space.read().unwrap();
        let from_addr = Addr::from(from.0);

        bind_space.edges_out(from_addr)
            .map(|e| SubstrateEdge {
                from: CogAddr::from(e.from.0),
                to: CogAddr::from(e.to.0),
                verb: CogAddr::from(e.verb.0),
                fingerprint: e.fingerprint,
                weight: e.weight,
                created: Instant::now(),
                lance_id: None,
                dirty: false,
            })
            .collect()
    }

    /// Get incoming edges
    pub fn edges_in(&self, to: CogAddr) -> Vec<SubstrateEdge> {
        let bind_space = self.bind_space.read().unwrap();
        let to_addr = Addr::from(to.0);

        bind_space.edges_in(to_addr)
            .map(|e| SubstrateEdge {
                from: CogAddr::from(e.from.0),
                to: CogAddr::from(e.to.0),
                verb: CogAddr::from(e.verb.0),
                fingerprint: e.fingerprint,
                weight: e.weight,
                created: Instant::now(),
                lance_id: None,
                dirty: false,
            })
            .collect()
    }

    /// Traverse via verb
    pub fn traverse(&self, from: CogAddr, verb: CogAddr) -> Vec<CogAddr> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space.traverse(Addr::from(from.0), Addr::from(verb.0))
            .into_iter()
            .map(|a| CogAddr::from(a.0))
            .collect()
    }

    /// N-hop traversal
    pub fn traverse_n_hops(&self, start: CogAddr, verb: CogAddr, max_hops: usize) -> Vec<(usize, CogAddr)> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space.traverse_n_hops(Addr::from(start.0), Addr::from(verb.0), max_hops)
            .into_iter()
            .map(|(hop, a)| (hop, CogAddr::from(a.0)))
            .collect()
    }

    // =========================================================================
    // SEARCH OPERATIONS
    // =========================================================================

    /// Search for similar nodes
    pub fn search(&self, query_fp: &[u64; FINGERPRINT_WORDS], k: usize, threshold: f32) -> Vec<(CogAddr, u32, f32)> {
        let bind_space = self.bind_space.read().unwrap();
        let mut results = Vec::new();

        // Scan node space (0x80-0xFF)
        for prefix in 0x80..=0xFF_u8 {
            for slot in 0..=255u8 {
                let addr = Addr::new(prefix, slot);
                if let Some(node) = bind_space.read(addr) {
                    let dist = hamming_distance(query_fp, &node.fingerprint);
                    let sim = 1.0 - (dist as f32 / 10000.0);
                    if sim >= threshold {
                        results.push((CogAddr::from_parts(prefix, slot), dist, sim));
                    }
                }
            }
        }

        // Sort by distance
        results.sort_by_key(|(_, d, _)| *d);
        results.truncate(k);

        results
    }

    /// Resonate: find nodes that resonate with query
    pub fn resonate(&self, query_fp: &[u64; FINGERPRINT_WORDS], k: usize) -> Vec<(CogAddr, f32)> {
        self.search(query_fp, k, 0.0)
            .into_iter()
            .map(|(addr, _, sim)| (addr, sim))
            .collect()
    }

    // =========================================================================
    // LIFECYCLE OPERATIONS
    // =========================================================================

    /// Tick: expire old fluid entries, promote/demote as needed
    pub fn tick(&self) {
        let mut extended = self.extended_nodes.write().unwrap();
        let mut to_remove = Vec::new();
        let mut to_promote = Vec::new();

        for (&id, node) in extended.iter() {
            if node.is_expired() {
                to_remove.push(id);
            } else if node.should_promote(self.config.promotion_threshold) {
                to_promote.push(id);
            }
        }

        // Evict expired
        for id in to_remove {
            extended.remove(&id);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }

        // Promote hot entries
        for id in to_promote {
            if let Some(mut node) = extended.remove(&id) {
                node.ttl = None;  // Now permanent
                // Write to node space
                let mut bind_space = self.bind_space.write().unwrap();
                let addr = bind_space.write_labeled(node.fingerprint, node.label.as_deref().unwrap_or(""));
                node.addr = CogAddr::from(addr.0);
                drop(bind_space);

                self.stats.promotions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Flush write buffer to Lance (sync version)
    pub fn flush_sync(&self) {
        let mut buffer = self.write_buffer.lock().unwrap();
        let ops = buffer.drain();

        if ops.is_empty() {
            return;
        }

        // In production, this would batch write to Lance
        // For now, just update stats
        self.stats.lance_writes.fetch_add(ops.len() as u64, Ordering::Relaxed);
        self.stats.pending_writes.store(0, Ordering::Relaxed);
    }

    /// Commit: flush all pending writes
    pub fn commit(&self) {
        self.flush_sync();
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    // =========================================================================
    // SURFACE ACCESS (CAM operations)
    // =========================================================================

    /// Get a verb by name
    pub fn verb(&self, name: &str) -> Option<CogAddr> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space.verb(name).map(|a| CogAddr::from(a.0))
    }

    /// Get a surface operation by compartment and name
    pub fn surface_op(&self, compartment: u8, name: &str) -> Option<CogAddr> {
        let bind_space = self.bind_space.read().unwrap();
        bind_space.surface_op(compartment, name).map(|a| CogAddr::from(a.0))
    }

    // =========================================================================
    // STATISTICS
    // =========================================================================

    /// Get statistics
    pub fn stats(&self) -> &SubstrateStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &SubstrateConfig {
        &self.config
    }

    /// Current version
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::SeqCst)
    }

    /// Is Lance connected?
    pub fn is_lance_connected(&self) -> bool {
        self.lance_connected.load(Ordering::Relaxed)
    }
}

impl Default for Substrate {
    fn default() -> Self {
        Self::default_new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_write_read() {
        let substrate = Substrate::default_new();

        let fp = [42u64; FINGERPRINT_WORDS];
        let addr = substrate.write(fp);

        assert!(addr.is_node());

        let node = substrate.read(addr);
        assert!(node.is_some());
        assert_eq!(node.unwrap().fingerprint, fp);
    }

    #[test]
    fn test_substrate_labeled() {
        let substrate = Substrate::default_new();

        let fp = [1u64; FINGERPRINT_WORDS];
        let addr = substrate.write_labeled(fp, "test_node");

        let node = substrate.read(addr).unwrap();
        assert_eq!(node.label.as_deref(), Some("test_node"));
    }

    #[test]
    fn test_substrate_link() {
        let substrate = Substrate::default_new();

        let a = substrate.write_labeled([1u64; FINGERPRINT_WORDS], "A");
        let b = substrate.write_labeled([2u64; FINGERPRINT_WORDS], "B");

        let causes = substrate.verb("CAUSES").unwrap();
        let edge = substrate.link(a, causes, b);

        assert!(edge.is_some());

        let targets = substrate.traverse(a, causes);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0], b);
    }

    #[test]
    fn test_substrate_search() {
        let substrate = Substrate::default_new();

        // Write some nodes with known fingerprints
        let fp1 = [0xAAAAAAAAAAAAAAAAu64; FINGERPRINT_WORDS];
        let fp2 = [0xBBBBBBBBBBBBBBBBu64; FINGERPRINT_WORDS];
        let fp3 = [0xAAAAAAAAAAAABBBBu64; FINGERPRINT_WORDS];  // Similar to fp1

        substrate.write_labeled(fp1, "node1");
        substrate.write_labeled(fp2, "node2");
        substrate.write_labeled(fp3, "node3");

        // Search for fp1-like nodes
        let results = substrate.search(&fp1, 10, 0.5);

        // Should find at least node1 (exact match)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_substrate_fluid_zone() {
        let substrate = Substrate::default_new();

        let fp = [99u64; FINGERPRINT_WORDS];
        let addr = substrate.write_fluid(fp, Duration::from_millis(100));

        // Should be readable
        let node = substrate.read(addr);
        assert!(node.is_some());

        // After TTL, should expire on tick
        std::thread::sleep(Duration::from_millis(150));
        substrate.tick();

        // Extended node should be gone
        let extended = substrate.extended_nodes.read().unwrap();
        assert!(!extended.contains_key(&(addr.0 as u64)));
    }

    #[test]
    fn test_substrate_stats() {
        let substrate = Substrate::default_new();

        let fp = [1u64; FINGERPRINT_WORDS];
        let addr = substrate.write(fp);

        // Read should hit
        substrate.read(addr);
        assert_eq!(substrate.stats.hot_hits.load(Ordering::Relaxed), 1);

        // Read non-existent should miss
        substrate.read(CogAddr::new(0xFFFF));
        assert_eq!(substrate.stats.hot_misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_substrate_commit() {
        let substrate = Substrate::default_new();

        for i in 0..10 {
            let fp = [i as u64; FINGERPRINT_WORDS];
            substrate.write(fp);
        }

        assert!(substrate.stats.pending_writes.load(Ordering::Relaxed) > 0);

        substrate.commit();

        // Version should increment
        assert_eq!(substrate.version(), 2);
    }
}
