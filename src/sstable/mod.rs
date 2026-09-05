//! SSTable on-disk format core: layout constants, magic bytes, and the
//! tri-state lookup result consumed by readers.

pub(crate) mod bloom;
pub mod builder;
pub(crate) mod index;
pub mod reader;
pub(crate) mod record;

pub const TARGET_BLOCK_SIZE: usize = 4096;
pub const MAGIC: [u8; 8] = [0x4C, 0x53, 0x4D, 0x54, 0x52, 0x45, 0x45, 0x31]; // "LSMTREE1"
pub const FOOTER_LEN: usize = 24;

pub(crate) const MAX_KEY_LEN: usize = 65535;
pub(crate) const MAX_VALUE_LEN: usize = u32::MAX as usize;

/// Bit0 of a record's flags byte marks a tombstone (deleted key). Upper flag
/// bits are reserved for future use and ignored on read.
pub const FLAG_DELETED: u8 = 0b0000_0001;

/// Result of a point lookup: a live value, or a tombstone (Deleted). An empty
/// `Present` value is legitimate user data and must be distinguishable from a
/// deletion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LookupValue {
    Present(Vec<u8>),
    Deleted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_bytes() {
        assert_eq!(MAGIC, *b"LSMTREE1");
    }

    #[test]
    fn lookup_value_variants() {
        assert_ne!(LookupValue::Present(b"x".to_vec()), LookupValue::Deleted);
        assert_ne!(LookupValue::Deleted, LookupValue::Present(Vec::new()));
        assert_eq!(
            LookupValue::Present(Vec::new()),
            LookupValue::Present(Vec::new())
        );
    }
}
