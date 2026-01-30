//! Optional Extensions for LadybugDB
//!
//! Enable via Cargo features: `codebook`, `hologram`, `spo`, `compress`

#[cfg(feature = "codebook")]
pub mod codebook;

#[cfg(feature = "hologram")]
pub mod hologram;

#[cfg(feature = "spo")]
pub mod spo;

#[cfg(feature = "spo")]
pub mod context_crystal;

#[cfg(feature = "spo")]
pub mod meta_resonance;

#[cfg(feature = "compress")]
pub mod compress;
