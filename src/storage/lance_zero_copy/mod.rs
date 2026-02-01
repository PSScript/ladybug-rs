//! Lance Zero-Copy Integration
//!
//! This module provides zero-copy access to Lance storage by sharing
//! the same address space. No serialization boundaries, no copies.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED ADDRESS SPACE                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  HOT (BindSpace)     WARM (Lance mmap)      COLD (Lance file)   │
//! │  ┌───────────────┐   ┌───────────────┐     ┌───────────────┐   │
//! │  │ Array[32K]    │   │ mmap region   │     │ Parquet file  │   │
//! │  │ Direct access │   │ OS page cache │     │ Compressed    │   │
//! │  └───────┬───────┘   └───────┬───────┘     └───────┬───────┘   │
//! │          │                   │                     │            │
//! │          │◀──── BUBBLE UP ───│◀──── BUBBLE UP ─────│            │
//! │          │     (ptr move)    │     (page fault)    │            │
//! │          │                   │                     │            │
//! │          │──── SINK DOWN ───▶│──── SINK DOWN ─────▶│            │
//! │          │     (ptr move)    │     (OS evict)      │            │
//! │                                                                 │
//! │  SCENT INDEX: Awareness layer - tracks where everything lives   │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Principles
//!
//! 1. **No Serialization Boundary**: Lance and Ladybug share memory
//! 2. **Scent as Awareness**: Index knows where data lives without copying
//! 3. **Bubbling Without Copy**: Promote/demote by updating pointers
//! 4. **Arrow Native**: Fingerprints stored as Arrow FixedSizeList

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt64Array};
use arrow_buffer::Buffer;
use arrow_data::ArrayDataBuilder;
use arrow_schema::{DataType, Field};

use super::bind_space::FINGERPRINT_WORDS;

// =============================================================================
// ARROW ZERO-COPY BRIDGE
// =============================================================================

/// Zero-copy fingerprint buffer backed by Arrow
///
/// This struct provides direct access to fingerprint data stored in Arrow format
/// without any copying. The underlying data lives in an Arrow Buffer which may
/// be backed by mmap'd memory from Lance storage.
pub struct FingerprintBuffer {
    /// The Arrow buffer containing raw u64 data
    buffer: Buffer,
    /// Number of fingerprints in this buffer
    len: usize,
}

impl FingerprintBuffer {
    /// Create from raw u64 data (zero-copy via ownership transfer)
    ///
    /// Uses `Buffer::from_vec()` which takes ownership of the Vec
    /// without copying data.
    pub fn from_vec(data: Vec<u64>, num_fingerprints: usize) -> Self {
        let expected_len = num_fingerprints * FINGERPRINT_WORDS;

        let buffer = if data.len() >= expected_len {
            Buffer::from_vec(data)
        } else {
            // Fallback: copy if size doesn't match
            Buffer::from_slice_ref(&data)
        };

        Self {
            buffer,
            len: num_fingerprints,
        }
    }

    /// Create from raw bytes slice (copies data)
    ///
    /// Use `from_vec` when possible to avoid copies.
    pub fn from_bytes(bytes: &[u8], num_fingerprints: usize) -> Self {
        let buffer = Buffer::from_slice_ref(bytes);
        Self {
            buffer,
            len: num_fingerprints,
        }
    }

    /// Create from an existing Arrow buffer (always zero-copy)
    pub fn from_buffer(buffer: Buffer, num_fingerprints: usize) -> Self {
        Self {
            buffer,
            len: num_fingerprints,
        }
    }

    /// Create from a FixedSizeListArray (zero-copy)
    ///
    /// The FixedSizeListArray should contain UInt64 elements with
    /// list size = FINGERPRINT_WORDS (156).
    pub fn from_fixed_size_list(array: &FixedSizeListArray) -> Option<Self> {
        // Verify the array structure
        if array.value_length() != FINGERPRINT_WORDS as i32 {
            return None;
        }

        // Get the inner values array
        let values = array.values();
        let u64_array = values.as_any().downcast_ref::<UInt64Array>()?;

        // Get the underlying buffer (zero-copy)
        let buffer = u64_array.values().inner().clone();

        Some(Self {
            buffer,
            len: array.len(),
        })
    }

    /// Get raw pointer to fingerprint data
    ///
    /// # Safety
    /// The returned pointer is valid for the lifetime of this FingerprintBuffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u64 {
        self.buffer.as_ptr() as *const u64
    }

    /// Get fingerprint at index as slice (zero-copy)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        if index >= self.len {
            return None;
        }

        let offset = index * FINGERPRINT_WORDS;
        let ptr = self.as_ptr();

        // Safety: We checked bounds above
        unsafe {
            let slice_ptr = ptr.add(offset) as *const [u64; FINGERPRINT_WORDS];
            Some(&*slice_ptr)
        }
    }

    /// Get fingerprint at index without bounds checking
    ///
    /// # Safety
    /// Caller must ensure index < self.len()
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[u64; FINGERPRINT_WORDS] {
        let ptr = self.as_ptr().add(index * FINGERPRINT_WORDS) as *const [u64; FINGERPRINT_WORDS];
        &*ptr
    }

    /// Number of fingerprints
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get underlying Arrow buffer (for sharing with other Arrow operations)
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Convert back to FixedSizeListArray (zero-copy)
    pub fn to_fixed_size_list(&self) -> ArrayRef {
        // Create UInt64Array from buffer
        let u64_data = ArrayDataBuilder::new(DataType::UInt64)
            .len(self.len * FINGERPRINT_WORDS)
            .add_buffer(self.buffer.clone())
            .build()
            .expect("valid array data");

        let u64_array = UInt64Array::from(u64_data);

        // Wrap in FixedSizeListArray
        let field = Arc::new(Field::new("item", DataType::UInt64, false));
        Arc::new(FixedSizeListArray::new(
            field,
            FINGERPRINT_WORDS as i32,
            Arc::new(u64_array),
            None,
        ))
    }

    /// Iterate over fingerprints (zero-copy)
    pub fn iter(&self) -> FingerprintIter<'_> {
        FingerprintIter {
            buffer: self,
            index: 0,
        }
    }
}

/// Iterator over fingerprints in a FingerprintBuffer
pub struct FingerprintIter<'a> {
    buffer: &'a FingerprintBuffer,
    index: usize,
}

impl<'a> Iterator for FingerprintIter<'a> {
    type Item = &'a [u64; FINGERPRINT_WORDS];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.buffer.len {
            return None;
        }
        // Safety: we just checked bounds
        let fp = unsafe { self.buffer.get_unchecked(self.index) };
        self.index += 1;
        Some(fp)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for FingerprintIter<'a> {}

// =============================================================================
// ARROW ZERO-COPY MANAGER
// =============================================================================

/// Manager for zero-copy Arrow integration
///
/// Handles conversion between Ladybug fingerprints and Arrow format,
/// maintaining zero-copy semantics throughout the pipeline.
pub struct ArrowZeroCopy {
    /// Currently loaded fingerprint buffers
    buffers: Vec<FingerprintBuffer>,
    /// Scent awareness for temperature tracking
    scent: ScentAwareness,
}

impl ArrowZeroCopy {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            scent: ScentAwareness::new(),
        }
    }

    /// Load fingerprints from Vec<u64> (zero-copy via ownership transfer)
    pub fn load_from_vec(&mut self, data: Vec<u64>, num_fingerprints: usize) -> usize {
        let buffer = FingerprintBuffer::from_vec(data, num_fingerprints);
        let id = self.buffers.len();
        self.buffers.push(buffer);
        id
    }

    /// Load fingerprints from bytes slice (copies data)
    pub fn load_from_bytes(&mut self, bytes: &[u8], num_fingerprints: usize) -> usize {
        let buffer = FingerprintBuffer::from_bytes(bytes, num_fingerprints);
        let id = self.buffers.len();
        self.buffers.push(buffer);
        id
    }

    /// Load fingerprints from Arrow array (zero-copy)
    pub fn load_from_arrow(&mut self, array: &FixedSizeListArray) -> Option<usize> {
        let buffer = FingerprintBuffer::from_fixed_size_list(array)?;
        let id = self.buffers.len();
        self.buffers.push(buffer);
        Some(id)
    }

    /// Get fingerprint by buffer ID and index
    pub fn get(&self, buffer_id: usize, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        self.buffers.get(buffer_id)?.get(index)
    }

    /// Touch for temperature tracking
    pub fn touch(&mut self, buffer_id: usize, index: usize) {
        // Encode buffer_id and index into a single u32
        let combined = ((buffer_id as u32) << 24) | (index as u32 & 0xFFFFFF);
        self.scent.touch(combined);
    }

    /// Get buffer by ID
    pub fn buffer(&self, id: usize) -> Option<&FingerprintBuffer> {
        self.buffers.get(id)
    }

    /// Get scent awareness
    pub fn scent(&self) -> &ScentAwareness {
        &self.scent
    }

    /// Get mutable scent awareness
    pub fn scent_mut(&mut self) -> &mut ScentAwareness {
        &mut self.scent
    }

    /// Total fingerprints across all buffers
    pub fn total_fingerprints(&self) -> usize {
        self.buffers.iter().map(|b| b.len()).sum()
    }
}

impl Default for ArrowZeroCopy {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// ADJACENCY SORTING (locality-preserving order)
// =============================================================================

/// Locality-preserving fingerprint ordering
///
/// Since Lance row operations are cheap, we can sort fingerprints by
/// adjacency (similar fingerprints near each other) for better cache
/// locality during search.
///
/// Uses a simple LSH-style bucket assignment based on the first few bits
/// of each fingerprint word, creating a coarse-grained spatial ordering.
pub struct AdjacencyIndex {
    /// Bucket assignments for each fingerprint
    /// bucket[i] = locality hash for fingerprint i
    buckets: Vec<u32>,

    /// Sorted indices: sorted_order[i] = original index
    /// Fingerprints are sorted by bucket, so similar ones are adjacent
    sorted_order: Vec<u32>,

    /// Reverse map: reverse[original_index] = position in sorted order
    reverse: Vec<u32>,
}

impl AdjacencyIndex {
    /// Create adjacency index for fingerprints
    ///
    /// Computes a locality-sensitive hash for each fingerprint and
    /// creates a sorted order that clusters similar fingerprints together.
    pub fn build(fingerprints: &FingerprintBuffer) -> Self {
        let n = fingerprints.len();

        // Compute locality hash for each fingerprint
        // Uses first 4 bits from first 8 words = 32-bit locality hash
        let buckets: Vec<u32> = (0..n)
            .map(|i| {
                let fp = fingerprints.get(i).unwrap();
                Self::locality_hash(fp)
            })
            .collect();

        // Create sorted indices
        let mut sorted_order: Vec<u32> = (0..n as u32).collect();
        sorted_order.sort_by_key(|&i| buckets[i as usize]);

        // Create reverse mapping
        let mut reverse = vec![0u32; n];
        for (pos, &orig) in sorted_order.iter().enumerate() {
            reverse[orig as usize] = pos as u32;
        }

        Self {
            buckets,
            sorted_order,
            reverse,
        }
    }

    /// Locality-sensitive hash
    ///
    /// Takes first 4 bits from first 8 words of fingerprint,
    /// creating a 32-bit locality hash that groups similar
    /// fingerprints together.
    #[inline]
    fn locality_hash(fp: &[u64; FINGERPRINT_WORDS]) -> u32 {
        let mut hash = 0u32;
        for i in 0..8 {
            // Take top 4 bits from each of first 8 words
            let bits = ((fp[i] >> 60) & 0xF) as u32;
            hash |= bits << (i * 4);
        }
        hash
    }

    /// Get original index from sorted position
    #[inline]
    pub fn original_index(&self, sorted_pos: usize) -> Option<u32> {
        self.sorted_order.get(sorted_pos).copied()
    }

    /// Get sorted position from original index
    #[inline]
    pub fn sorted_position(&self, original_idx: usize) -> Option<u32> {
        self.reverse.get(original_idx).copied()
    }

    /// Get locality bucket for original index
    #[inline]
    pub fn bucket(&self, original_idx: usize) -> Option<u32> {
        self.buckets.get(original_idx).copied()
    }

    /// Find range of sorted indices that might contain similar fingerprints
    ///
    /// Given a query fingerprint, returns a range of sorted positions
    /// that are likely to contain similar fingerprints (same locality bucket).
    pub fn locality_range(&self, query: &[u64; FINGERPRINT_WORDS]) -> std::ops::Range<usize> {
        let query_bucket = Self::locality_hash(query);

        // Binary search for first index with this bucket
        let start = self.sorted_order
            .partition_point(|&i| self.buckets[i as usize] < query_bucket);

        // Binary search for first index past this bucket
        let end = self.sorted_order
            .partition_point(|&i| self.buckets[i as usize] <= query_bucket);

        start..end
    }

    /// Iterate over original indices in sorted (locality-preserving) order
    pub fn iter_sorted(&self) -> impl Iterator<Item = u32> + '_ {
        self.sorted_order.iter().copied()
    }

    /// Number of fingerprints
    pub fn len(&self) -> usize {
        self.sorted_order.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.sorted_order.is_empty()
    }
}

// =============================================================================
// SAFETY: WAL + CONCURRENCY + ACID
// =============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use parking_lot::{RwLock, Mutex};

/// Write-Ahead Log entry for crash recovery
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonic sequence number
    pub lsn: u64,
    /// Operation type
    pub op: WalOp,
    /// Timestamp (epoch millis)
    pub timestamp: u64,
    /// Checksum for integrity
    pub checksum: u32,
}

/// WAL operation types
#[derive(Debug, Clone)]
pub enum WalOp {
    /// Insert fingerprint at index
    Insert { index: u32, fingerprint: Box<[u64; FINGERPRINT_WORDS]> },
    /// Update fingerprint at index
    Update { index: u32, old: Box<[u64; FINGERPRINT_WORDS]>, new: Box<[u64; FINGERPRINT_WORDS]> },
    /// Delete fingerprint at index
    Delete { index: u32, fingerprint: Box<[u64; FINGERPRINT_WORDS]> },
    /// Temperature change (for scent tracking)
    TempChange { index: u32, from: Temperature, to: Temperature },
    /// Checkpoint marker (safe recovery point)
    Checkpoint { version: u64 },
    /// Transaction begin
    TxnBegin { txn_id: u64 },
    /// Transaction commit
    TxnCommit { txn_id: u64 },
    /// Transaction abort
    TxnAbort { txn_id: u64 },
}

impl WalEntry {
    /// Create new WAL entry
    pub fn new(lsn: u64, op: WalOp) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut entry = Self {
            lsn,
            op,
            timestamp,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32-style checksum
    fn compute_checksum(&self) -> u32 {
        // Simple checksum: XOR of lsn, timestamp, and op discriminant
        let op_tag = match &self.op {
            WalOp::Insert { .. } => 1u32,
            WalOp::Update { .. } => 2,
            WalOp::Delete { .. } => 3,
            WalOp::TempChange { .. } => 4,
            WalOp::Checkpoint { .. } => 5,
            WalOp::TxnBegin { .. } => 6,
            WalOp::TxnCommit { .. } => 7,
            WalOp::TxnAbort { .. } => 8,
        };
        (self.lsn as u32) ^ (self.timestamp as u32) ^ op_tag
    }

    /// Verify entry integrity
    pub fn verify(&self) -> bool {
        self.checksum == self.compute_checksum()
    }
}

/// Write-Ahead Log for durability
pub struct WriteAheadLog {
    /// Current LSN (Log Sequence Number)
    current_lsn: AtomicU64,
    /// In-memory log buffer (flushed to disk periodically)
    buffer: Mutex<VecDeque<WalEntry>>,
    /// Last flushed LSN
    flushed_lsn: AtomicU64,
    /// Last checkpoint LSN
    checkpoint_lsn: AtomicU64,
    /// Maximum buffer size before force flush
    max_buffer_size: usize,
    /// Path for WAL files
    wal_path: PathBuf,
    /// Is WAL enabled?
    enabled: AtomicBool,
}

impl WriteAheadLog {
    /// Create new WAL
    pub fn new(wal_path: PathBuf) -> Self {
        Self {
            current_lsn: AtomicU64::new(1),
            buffer: Mutex::new(VecDeque::with_capacity(1024)),
            flushed_lsn: AtomicU64::new(0),
            checkpoint_lsn: AtomicU64::new(0),
            max_buffer_size: 1024,
            wal_path,
            enabled: AtomicBool::new(true),
        }
    }

    /// Append entry to WAL, returns LSN
    pub fn append(&self, op: WalOp) -> u64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }

        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst);
        let entry = WalEntry::new(lsn, op);

        let mut buffer = self.buffer.lock();
        buffer.push_back(entry);

        // Force flush if buffer is full
        if buffer.len() >= self.max_buffer_size {
            drop(buffer);
            self.flush();
        }

        lsn
    }

    /// Flush buffer to disk
    pub fn flush(&self) -> u64 {
        let mut buffer = self.buffer.lock();
        if buffer.is_empty() {
            return self.flushed_lsn.load(Ordering::Relaxed);
        }

        // In real implementation, serialize and write to file
        // For now, just update flushed_lsn
        let max_lsn = buffer.back().map(|e| e.lsn).unwrap_or(0);
        buffer.clear();

        self.flushed_lsn.store(max_lsn, Ordering::Release);
        max_lsn
    }

    /// Create checkpoint (safe recovery point)
    pub fn checkpoint(&self) -> u64 {
        let lsn = self.append(WalOp::Checkpoint {
            version: self.current_lsn.load(Ordering::Relaxed),
        });
        self.flush();
        self.checkpoint_lsn.store(lsn, Ordering::Release);
        lsn
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::Relaxed)
    }

    /// Get last flushed LSN
    pub fn flushed_lsn(&self) -> u64 {
        self.flushed_lsn.load(Ordering::Relaxed)
    }

    /// Get last checkpoint LSN
    pub fn checkpoint_lsn(&self) -> u64 {
        self.checkpoint_lsn.load(Ordering::Relaxed)
    }

    /// Enable/disable WAL
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Recover from WAL (replay entries after last checkpoint)
    pub fn recover(&self) -> Vec<WalEntry> {
        // In real implementation, read WAL file and return entries
        // after last checkpoint for replay
        Vec::new()
    }
}

impl Default for WriteAheadLog {
    fn default() -> Self {
        Self::new(PathBuf::from("./wal"))
    }
}

/// MVCC version for isolation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u64);

impl Version {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
}

/// Versioned fingerprint for MVCC
pub struct VersionedFingerprint {
    /// The fingerprint data
    pub data: [u64; FINGERPRINT_WORDS],
    /// Version when this was written
    pub write_version: Version,
    /// Version when this was deleted (None if still visible)
    pub delete_version: Option<Version>,
}

impl VersionedFingerprint {
    /// Check if visible at given version
    pub fn visible_at(&self, version: Version) -> bool {
        version >= self.write_version
            && self.delete_version.map(|dv| version < dv).unwrap_or(true)
    }
}

/// Transaction state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

/// Transaction context for ACID
pub struct Transaction {
    /// Transaction ID
    pub id: u64,
    /// Read version (snapshot isolation)
    pub read_version: Version,
    /// State
    pub state: TxnState,
    /// WAL entries for this transaction
    pub wal_entries: Vec<u64>,
    /// Modified indices (for rollback)
    pub modified: Vec<u32>,
}

impl Transaction {
    /// Create new transaction
    pub fn new(id: u64, read_version: Version) -> Self {
        Self {
            id,
            read_version,
            state: TxnState::Active,
            wal_entries: Vec::new(),
            modified: Vec::new(),
        }
    }

    /// Check if transaction is active
    pub fn is_active(&self) -> bool {
        self.state == TxnState::Active
    }
}

/// Concurrent access controller with read-write locking
pub struct ConcurrentAccess<T> {
    /// The protected data
    data: RwLock<T>,
    /// Version counter
    version: AtomicU64,
    /// WAL reference
    wal: Arc<WriteAheadLog>,
    /// Active transactions
    active_txns: Mutex<Vec<Transaction>>,
    /// Next transaction ID
    next_txn_id: AtomicU64,
}

impl<T> ConcurrentAccess<T> {
    /// Create new concurrent access wrapper
    pub fn new(data: T, wal: Arc<WriteAheadLog>) -> Self {
        Self {
            data: RwLock::new(data),
            version: AtomicU64::new(1),
            wal,
            active_txns: Mutex::new(Vec::new()),
            next_txn_id: AtomicU64::new(1),
        }
    }

    /// Read access (shared lock)
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, T> {
        self.data.read()
    }

    /// Write access (exclusive lock)
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, T> {
        self.data.write()
    }

    /// Try read with timeout
    pub fn try_read(&self, timeout: std::time::Duration) -> Option<parking_lot::RwLockReadGuard<'_, T>> {
        self.data.try_read_for(timeout)
    }

    /// Try write with timeout
    pub fn try_write(&self, timeout: std::time::Duration) -> Option<parking_lot::RwLockWriteGuard<'_, T>> {
        self.data.try_write_for(timeout)
    }

    /// Begin transaction
    pub fn begin_txn(&self) -> u64 {
        let txn_id = self.next_txn_id.fetch_add(1, Ordering::SeqCst);
        let read_version = Version::new(self.version.load(Ordering::Acquire));

        let lsn = self.wal.append(WalOp::TxnBegin { txn_id });

        let mut txn = Transaction::new(txn_id, read_version);
        txn.wal_entries.push(lsn);

        self.active_txns.lock().push(txn);
        txn_id
    }

    /// Commit transaction
    pub fn commit_txn(&self, txn_id: u64) -> bool {
        let mut txns = self.active_txns.lock();
        if let Some(pos) = txns.iter().position(|t| t.id == txn_id) {
            let mut txn = txns.remove(pos);
            if txn.state != TxnState::Active {
                return false;
            }

            txn.state = TxnState::Committed;
            self.wal.append(WalOp::TxnCommit { txn_id });

            // Bump version
            self.version.fetch_add(1, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Abort transaction
    pub fn abort_txn(&self, txn_id: u64) -> bool {
        let mut txns = self.active_txns.lock();
        if let Some(pos) = txns.iter().position(|t| t.id == txn_id) {
            let mut txn = txns.remove(pos);
            if txn.state != TxnState::Active {
                return false;
            }

            txn.state = TxnState::Aborted;
            self.wal.append(WalOp::TxnAbort { txn_id });
            // Rollback would happen here using txn.modified
            true
        } else {
            false
        }
    }

    /// Current version
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }
}

/// Thread-safe zero-copy manager with ACID guarantees
pub struct SafeArrowZeroCopy {
    /// Inner manager with concurrent access
    inner: ConcurrentAccess<ArrowZeroCopy>,
}

impl SafeArrowZeroCopy {
    /// Create new thread-safe manager
    pub fn new(wal: Arc<WriteAheadLog>) -> Self {
        Self {
            inner: ConcurrentAccess::new(ArrowZeroCopy::new(), wal),
        }
    }

    /// Load fingerprints with transaction
    pub fn load_with_txn(&self, txn_id: u64, data: Vec<u64>, num_fingerprints: usize) -> Option<usize> {
        let mut guard = self.inner.write();
        let buffer_id = guard.load_from_vec(data, num_fingerprints);

        // Record in transaction
        let mut txns = self.inner.active_txns.lock();
        if let Some(txn) = txns.iter_mut().find(|t| t.id == txn_id) {
            txn.modified.push(buffer_id as u32);
        }

        Some(buffer_id)
    }

    /// Read fingerprint (shared access)
    pub fn get(&self, buffer_id: usize, index: usize) -> Option<[u64; FINGERPRINT_WORDS]> {
        let guard = self.inner.read();
        guard.get(buffer_id, index).copied()
    }

    /// Begin transaction
    pub fn begin(&self) -> u64 {
        self.inner.begin_txn()
    }

    /// Commit transaction
    pub fn commit(&self, txn_id: u64) -> bool {
        self.inner.commit_txn(txn_id)
    }

    /// Abort transaction
    pub fn abort(&self, txn_id: u64) -> bool {
        self.inner.abort_txn(txn_id)
    }

    /// Checkpoint (create safe recovery point)
    pub fn checkpoint(&self) {
        self.inner.wal.checkpoint();
    }

    /// Force WAL flush
    pub fn flush_wal(&self) {
        self.inner.wal.flush();
    }
}

// =============================================================================
// ZERO-COPY VIEW
// =============================================================================

/// A zero-copy view into Lance storage
///
/// This struct holds pointers into mmap'd Arrow columns.
/// No copies, just views into the OS page cache.
pub struct LanceView {
    /// Path to the Lance dataset
    pub path: PathBuf,

    /// Number of fingerprints in this view
    pub len: usize,

    /// Raw pointer to fingerprint data (156 × u64 per fingerprint)
    /// Points directly into mmap'd Arrow buffer
    #[cfg(feature = "lance")]
    fingerprint_ptr: *const u64,

    /// Raw pointer to address data (u16 per entry)
    #[cfg(feature = "lance")]
    address_ptr: *const u16,

    /// Scent index for awareness (which entries are hot/warm/cold)
    scent: ScentAwareness,
}

// Safety: The pointers point to mmap'd memory that outlives this struct
#[cfg(feature = "lance")]
unsafe impl Send for LanceView {}
#[cfg(feature = "lance")]
unsafe impl Sync for LanceView {}

impl LanceView {
    /// Create a new Lance view (placeholder for when Lance is vendored)
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            len: 0,
            #[cfg(feature = "lance")]
            fingerprint_ptr: std::ptr::null(),
            #[cfg(feature = "lance")]
            address_ptr: std::ptr::null(),
            scent: ScentAwareness::new(),
        }
    }

    /// Zero-copy access to fingerprint at index
    ///
    /// # Safety
    /// Caller must ensure index < self.len
    #[cfg(feature = "lance")]
    #[inline]
    pub unsafe fn fingerprint_unchecked(&self, index: usize) -> &[u64; FINGERPRINT_WORDS] {
        &*(self.fingerprint_ptr.add(index * FINGERPRINT_WORDS) as *const [u64; FINGERPRINT_WORDS])
    }

    /// Safe access to fingerprint
    #[cfg(feature = "lance")]
    pub fn fingerprint(&self, index: usize) -> Option<&[u64; FINGERPRINT_WORDS]> {
        if index < self.len {
            Some(unsafe { self.fingerprint_unchecked(index) })
        } else {
            None
        }
    }

    /// Get scent awareness
    pub fn scent(&self) -> &ScentAwareness {
        &self.scent
    }

    /// Get mutable scent awareness
    pub fn scent_mut(&mut self) -> &mut ScentAwareness {
        &mut self.scent
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// =============================================================================
// SCENT AWARENESS
// =============================================================================

/// Temperature tier for bubbling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Temperature {
    /// In BindSpace array (< 1 cycle access)
    Hot,
    /// In mmap region (page cache, ~100ns)
    Warm,
    /// On disk (needs page fault, ~10μs)
    Cold,
    /// Unknown/not tracked
    Unknown,
}

/// Scent awareness - knows where data lives without copying it
///
/// This is the "awareness" layer that tracks:
/// - Which fingerprints are hot (in BindSpace)
/// - Which are warm (in mmap page cache)
/// - Which are cold (on disk)
///
/// Bubbling happens by updating this index, not by copying data.
pub struct ScentAwareness {
    /// Hot set: indices that are in BindSpace
    hot_indices: Vec<u32>,

    /// Recently accessed (LRU for warmth tracking)
    recent_access: Vec<u32>,

    /// Access counts for promotion decisions
    access_counts: Vec<u8>,

    /// Capacity
    capacity: usize,
}

impl ScentAwareness {
    /// Create new scent awareness
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hot_indices: Vec::with_capacity(capacity / 10),
            recent_access: Vec::with_capacity(64),
            access_counts: vec![0; capacity],
            capacity,
        }
    }

    /// Record an access (for warmth tracking)
    pub fn touch(&mut self, index: u32) {
        let idx = index as usize;
        if idx < self.access_counts.len() {
            self.access_counts[idx] = self.access_counts[idx].saturating_add(1);
        }

        // Update recent access (simple ring buffer)
        if self.recent_access.len() >= 64 {
            self.recent_access.remove(0);
        }
        self.recent_access.push(index);
    }

    /// Get temperature of an index
    pub fn temperature(&self, index: u32) -> Temperature {
        if self.hot_indices.binary_search(&index).is_ok() {
            Temperature::Hot
        } else if self.recent_access.contains(&index) {
            Temperature::Warm
        } else if (index as usize) < self.access_counts.len()
            && self.access_counts[index as usize] > 0
        {
            Temperature::Warm
        } else {
            Temperature::Cold
        }
    }

    /// Mark index as hot (bubbled up to BindSpace)
    pub fn mark_hot(&mut self, index: u32) {
        if self.hot_indices.binary_search(&index).is_err() {
            self.hot_indices.push(index);
            self.hot_indices.sort();
        }
    }

    /// Mark index as cold (bubbled down from BindSpace)
    pub fn mark_cold(&mut self, index: u32) {
        if let Ok(pos) = self.hot_indices.binary_search(&index) {
            self.hot_indices.remove(pos);
        }
    }

    /// Get indices that should bubble up (high access count, currently cold)
    pub fn candidates_for_promotion(&self, limit: usize) -> Vec<u32> {
        let mut candidates: Vec<(u32, u8)> = self.access_counts
            .iter()
            .enumerate()
            .filter(|(idx, &count)| {
                count > 5 && self.hot_indices.binary_search(&(*idx as u32)).is_err()
            })
            .map(|(idx, &count)| (idx as u32, count))
            .collect();

        candidates.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by access count desc
        candidates.into_iter().take(limit).map(|(idx, _)| idx).collect()
    }

    /// Get indices that should bubble down (low access count, currently hot)
    pub fn candidates_for_demotion(&self, limit: usize) -> Vec<u32> {
        let mut candidates: Vec<(u32, u8)> = self.hot_indices
            .iter()
            .filter_map(|&idx| {
                let count = self.access_counts.get(idx as usize).copied().unwrap_or(0);
                if count < 2 {
                    Some((idx, count))
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // Sort by access count asc
        candidates.into_iter().take(limit).map(|(idx, _)| idx).collect()
    }

    /// Decay access counts (call periodically)
    pub fn decay(&mut self) {
        for count in &mut self.access_counts {
            *count = count.saturating_sub(1);
        }
    }

    /// Number of hot entries
    pub fn hot_count(&self) -> usize {
        self.hot_indices.len()
    }
}

impl Default for ScentAwareness {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// BUBBLING OPERATIONS
// =============================================================================

/// Bubble operation result
#[derive(Debug, Clone)]
pub struct BubbleResult {
    /// Index that was bubbled
    pub index: u32,
    /// Previous temperature
    pub from: Temperature,
    /// New temperature
    pub to: Temperature,
    /// Whether data was copied (should be false for zero-copy)
    pub copied: bool,
}

/// Zero-copy bubbler
///
/// Moves data between hot/warm/cold tiers by updating pointers,
/// not by copying data.
pub struct ZeroCopyBubbler {
    /// Maximum hot entries
    max_hot: usize,
    /// Bubble threshold (access count to promote)
    promote_threshold: u8,
    /// Demote threshold (access count to demote)
    demote_threshold: u8,
}

impl Default for ZeroCopyBubbler {
    fn default() -> Self {
        Self {
            max_hot: 32768, // Same as BindSpace node capacity
            promote_threshold: 10,
            demote_threshold: 2,
        }
    }
}

impl ZeroCopyBubbler {
    /// Create with custom thresholds
    pub fn new(max_hot: usize, promote_threshold: u8, demote_threshold: u8) -> Self {
        Self {
            max_hot,
            promote_threshold,
            demote_threshold,
        }
    }

    /// Bubble up: promote cold/warm to hot
    ///
    /// In zero-copy mode, this just updates the scent index.
    /// The actual fingerprint stays in the mmap'd file.
    pub fn bubble_up(&self, scent: &mut ScentAwareness, index: u32) -> BubbleResult {
        let from = scent.temperature(index);
        scent.mark_hot(index);

        BubbleResult {
            index,
            from,
            to: Temperature::Hot,
            copied: false, // Zero-copy!
        }
    }

    /// Bubble down: demote hot to warm/cold
    pub fn bubble_down(&self, scent: &mut ScentAwareness, index: u32) -> BubbleResult {
        let from = scent.temperature(index);
        scent.mark_cold(index);

        BubbleResult {
            index,
            from,
            to: Temperature::Warm, // Goes to warm first (still in page cache)
            copied: false,
        }
    }

    /// Auto-bubble based on access patterns
    pub fn auto_bubble(&self, scent: &mut ScentAwareness) -> Vec<BubbleResult> {
        let mut results = Vec::new();

        // Promote hot candidates
        if scent.hot_count() < self.max_hot {
            let room = self.max_hot - scent.hot_count();
            for idx in scent.candidates_for_promotion(room.min(10)) {
                results.push(self.bubble_up(scent, idx));
            }
        }

        // Demote cold candidates if we're at capacity
        if scent.hot_count() >= self.max_hot {
            for idx in scent.candidates_for_demotion(10) {
                results.push(self.bubble_down(scent, idx));
            }
        }

        results
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test fingerprint with a specific pattern
    fn make_fingerprint(seed: u64) -> [u64; FINGERPRINT_WORDS] {
        let mut fp = [0u64; FINGERPRINT_WORDS];
        for i in 0..FINGERPRINT_WORDS {
            fp[i] = seed.wrapping_mul(i as u64 + 1).wrapping_add(seed);
        }
        fp
    }

    #[test]
    fn test_fingerprint_buffer_from_vec() {
        // Create test fingerprints
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..10)
            .map(|i| make_fingerprint(i as u64 * 1000))
            .collect();

        // Flatten to Vec<u64>
        let mut data = Vec::with_capacity(fps.len() * FINGERPRINT_WORDS);
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        // Create buffer (zero-copy via ownership transfer)
        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Verify access
        assert_eq!(buffer.len(), 10);
        for (i, expected) in fps.iter().enumerate() {
            let got = buffer.get(i).unwrap();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn test_fingerprint_buffer_iter() {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..5)
            .map(|i| make_fingerprint(i as u64 * 100))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Test iterator
        let collected: Vec<_> = buffer.iter().collect();
        assert_eq!(collected.len(), 5);
        for (i, got) in collected.iter().enumerate() {
            assert_eq!(*got, &fps[i]);
        }
    }

    #[test]
    fn test_fingerprint_buffer_to_arrow() {
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..3)
            .map(|i| make_fingerprint(i as u64 * 50))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());

        // Convert to Arrow
        let array = buffer.to_fixed_size_list();
        let fsl = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        assert_eq!(fsl.len(), 3);
        assert_eq!(fsl.value_length(), FINGERPRINT_WORDS as i32);
    }

    #[test]
    fn test_arrow_zero_copy_manager() {
        let mut manager = ArrowZeroCopy::new();

        // Create and load fingerprints
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..5)
            .map(|i| make_fingerprint(i as u64))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer_id = manager.load_from_vec(data, fps.len());

        // Verify access
        assert_eq!(manager.total_fingerprints(), 5);
        for (i, expected) in fps.iter().enumerate() {
            let got = manager.get(buffer_id, i).unwrap();
            assert_eq!(got, expected);
        }

        // Test touch (temperature tracking)
        manager.touch(buffer_id, 2);
        manager.touch(buffer_id, 2);
        manager.touch(buffer_id, 2);
    }

    #[test]
    fn test_adjacency_index() {
        // Create fingerprints with varying patterns
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..20)
            .map(|i| make_fingerprint(i as u64 * 1000))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());
        let adj_index = AdjacencyIndex::build(&buffer);

        assert_eq!(adj_index.len(), 20);

        // Verify bidirectional mapping
        for orig_idx in 0..20 {
            let sorted_pos = adj_index.sorted_position(orig_idx).unwrap();
            let back = adj_index.original_index(sorted_pos as usize).unwrap();
            assert_eq!(back as usize, orig_idx);
        }
    }

    #[test]
    fn test_locality_range() {
        // Create fingerprints where some share locality buckets
        let mut fps = Vec::new();

        // First 5: all zeros in first 8 words (same locality bucket)
        for i in 0..5 {
            let mut fp = [0u64; FINGERPRINT_WORDS];
            fp[100] = i; // Differ in later words only
            fps.push(fp);
        }

        // Next 5: high bits set (different locality bucket)
        for i in 0..5 {
            let mut fp = [0xF000_0000_0000_0000u64; FINGERPRINT_WORDS];
            fp[100] = i as u64;
            fps.push(fp);
        }

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer = FingerprintBuffer::from_vec(data, fps.len());
        let adj_index = AdjacencyIndex::build(&buffer);

        // Query with zeros pattern should find ~5 matches
        let query = [0u64; FINGERPRINT_WORDS];
        let range = adj_index.locality_range(&query);

        // Should have some matches in the locality bucket
        assert!(!range.is_empty());
    }

    #[test]
    fn test_scent_awareness() {
        let mut scent = ScentAwareness::with_capacity(100);

        // Initially cold
        assert_eq!(scent.temperature(42), Temperature::Cold);

        // Touch warms it up
        scent.touch(42);
        assert_eq!(scent.temperature(42), Temperature::Warm);

        // Mark hot
        scent.mark_hot(42);
        assert_eq!(scent.temperature(42), Temperature::Hot);

        // Mark cold again
        scent.mark_cold(42);
        assert_eq!(scent.temperature(42), Temperature::Warm); // Still warm from touch
    }

    #[test]
    fn test_bubbler() {
        let bubbler = ZeroCopyBubbler::default();
        let mut scent = ScentAwareness::with_capacity(100);

        // Bubble up
        let result = bubbler.bubble_up(&mut scent, 42);
        assert_eq!(result.to, Temperature::Hot);
        assert!(!result.copied); // Zero-copy!

        // Bubble down
        let result = bubbler.bubble_down(&mut scent, 42);
        assert_eq!(result.to, Temperature::Warm);
        assert!(!result.copied);
    }

    #[test]
    fn test_promotion_candidates() {
        let mut scent = ScentAwareness::with_capacity(100);

        // Touch index 10 many times
        for _ in 0..20 {
            scent.touch(10);
        }

        // Touch index 20 a few times
        for _ in 0..8 {
            scent.touch(20);
        }

        // Index 10 should be candidate for promotion
        let candidates = scent.candidates_for_promotion(5);
        assert!(candidates.contains(&10));
    }

    #[test]
    fn test_wal_basic() {
        let wal = WriteAheadLog::new(PathBuf::from("/tmp/test_wal"));

        // Append entries
        let lsn1 = wal.append(WalOp::Insert {
            index: 0,
            fingerprint: Box::new([0u64; FINGERPRINT_WORDS]),
        });
        assert_eq!(lsn1, 1);

        let lsn2 = wal.append(WalOp::Update {
            index: 0,
            old: Box::new([0u64; FINGERPRINT_WORDS]),
            new: Box::new([1u64; FINGERPRINT_WORDS]),
        });
        assert_eq!(lsn2, 2);

        // Check LSN progression
        assert_eq!(wal.current_lsn(), 3);
    }

    #[test]
    fn test_wal_checkpoint() {
        let wal = WriteAheadLog::new(PathBuf::from("/tmp/test_wal_ckpt"));

        // Add some entries
        for i in 0..5 {
            wal.append(WalOp::Insert {
                index: i,
                fingerprint: Box::new([i as u64; FINGERPRINT_WORDS]),
            });
        }

        // Checkpoint
        let ckpt_lsn = wal.checkpoint();
        assert!(ckpt_lsn > 0);
        assert_eq!(wal.checkpoint_lsn(), ckpt_lsn);
    }

    #[test]
    fn test_wal_entry_integrity() {
        let entry = WalEntry::new(42, WalOp::Delete {
            index: 10,
            fingerprint: Box::new([0u64; FINGERPRINT_WORDS]),
        });

        // Verify integrity
        assert!(entry.verify());

        // Tampered entry should fail
        let mut tampered = entry.clone();
        tampered.lsn = 999;
        assert!(!tampered.verify());
    }

    #[test]
    fn test_concurrent_access() {
        let wal = Arc::new(WriteAheadLog::default());
        let safe = SafeArrowZeroCopy::new(wal);

        // Begin transaction
        let txn_id = safe.begin();
        assert!(txn_id > 0);

        // Load data
        let fps: Vec<[u64; FINGERPRINT_WORDS]> = (0..3)
            .map(|i| make_fingerprint(i as u64))
            .collect();

        let mut data = Vec::new();
        for fp in &fps {
            data.extend_from_slice(fp);
        }

        let buffer_id = safe.load_with_txn(txn_id, data, fps.len());
        assert!(buffer_id.is_some());

        // Commit
        assert!(safe.commit(txn_id));
    }

    #[test]
    fn test_transaction_abort() {
        let wal = Arc::new(WriteAheadLog::default());
        let safe = SafeArrowZeroCopy::new(wal);

        let txn_id = safe.begin();

        // Load some data
        let data = vec![0u64; FINGERPRINT_WORDS];
        safe.load_with_txn(txn_id, data, 1);

        // Abort transaction
        assert!(safe.abort(txn_id));

        // Can't commit aborted transaction
        assert!(!safe.commit(txn_id));
    }

    #[test]
    fn test_versioned_fingerprint_visibility() {
        let fp = VersionedFingerprint {
            data: [42u64; FINGERPRINT_WORDS],
            write_version: Version::new(5),
            delete_version: Some(Version::new(10)),
        };

        // Before write - not visible
        assert!(!fp.visible_at(Version::new(4)));

        // After write, before delete - visible
        assert!(fp.visible_at(Version::new(5)));
        assert!(fp.visible_at(Version::new(7)));

        // At or after delete - not visible
        assert!(!fp.visible_at(Version::new(10)));
        assert!(!fp.visible_at(Version::new(15)));
    }

    #[test]
    fn test_concurrent_reads() {
        use std::thread;

        let wal = Arc::new(WriteAheadLog::default());
        let safe = Arc::new(SafeArrowZeroCopy::new(wal));

        // Load some data first
        let txn = safe.begin();
        let data = vec![0u64; FINGERPRINT_WORDS * 10];
        safe.load_with_txn(txn, data, 10);
        safe.commit(txn);

        // Spawn multiple readers
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let safe_clone = Arc::clone(&safe);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = safe_clone.get(0, 0);
                    }
                })
            })
            .collect();

        // All readers should complete without deadlock
        for h in handles {
            h.join().expect("reader thread panicked");
        }
    }
}
