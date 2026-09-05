//! Bloom filter block codec for sstables: seeded construction, manual
//! serialization (`[num_hashes u32 LE][words u64 LE x N]`), and reconstruction
//! from raw words.

use crate::error::{LsmError, Result};
use crate::utils;
use fastbloom::BloomFilter;

pub(crate) const BLOOM_FP_RATE: f64 = 0.01;
// KRITIS: fastbloom's `DefaultHasher::default()` is seeded randomly per
// instance (foldhash RandomState / rand). A bloom built without this fixed
// seed cannot be rebuilt deterministically from its raw words on decode, so
// the same constant MUST be applied at build time and rebuild time. Never
// change it without a file-format migration.
pub(crate) const BLOOM_SEED: u128 = 0x4C53_4D54_5245_4531_0000_0000_0000_0001;

/// Empty seeded bloom sized for `expected_items` (fastbloom floors at 64
/// bits). Always `.seed()` — see [`BLOOM_SEED`].
pub(crate) fn new_bloom(expected_items: usize) -> BloomFilter {
    BloomFilter::with_false_pos(BLOOM_FP_RATE)
        .seed(&BLOOM_SEED)
        .expected_items(expected_items)
}

/// Appends `[num_hashes u32 LE][words: u64 LE x N]` to `buf`.
pub(crate) fn encode_bloom(buf: &mut Vec<u8>, bloom: &BloomFilter) {
    utils::put_u32(buf, bloom.num_hashes());
    for word in bloom.iter() {
        utils::put_u64(buf, word);
    }
}

/// Decodes a bloom block: `Corruption` if `len < 4`, `(len - 4) % 8 != 0`,
/// the block has zero words, or the declared `num_hashes` is implausible.
/// `num_hashes` is capped: with `BLOOM_FP_RATE = 0.01` the optimal hash count
/// never exceeds ~44 (it peaks at the 64-bit floor), so 255 is a generous
/// upper bound that stops a hostile block from turning every `contains()` into
/// a multi-billion-iteration loop.
pub(crate) fn decode_bloom(bytes: &[u8]) -> Result<BloomFilter> {
    const MAX_NUM_HASHES: u32 = 255;
    if bytes.len() < 4 || !(bytes.len() - 4).is_multiple_of(8) {
        return Err(LsmError::Corruption(format!(
            "bloom block length {} is not 4 + a multiple of 8",
            bytes.len()
        )));
    }
    let num_hashes = utils::read_u32(bytes, 0)?;
    if num_hashes == 0 || num_hashes > MAX_NUM_HASHES {
        return Err(LsmError::Corruption(format!(
            "bloom block declares implausible hash count {num_hashes}"
        )));
    }
    let num_words = (bytes.len() - 4) / 8;
    if num_words == 0 {
        return Err(LsmError::Corruption(
            "bloom block has no data words".to_string(),
        ));
    }
    let words: Vec<u64> = (0..num_words)
        .map(|i| utils::read_u64(bytes, 4 + i * 8))
        .collect::<Result<_>>()?;
    Ok(BloomFilter::from_vec(words)
        .seed(&BLOOM_SEED)
        .hashes(num_hashes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_all_keys() {
        let mut bloom = new_bloom(1000);
        let keys: Vec<String> = (0..1000).map(|i| format!("k{i}")).collect();
        for k in &keys {
            bloom.insert(k.as_bytes());
        }
        let mut buf = Vec::new();
        encode_bloom(&mut buf, &bloom);
        let decoded = decode_bloom(&buf).expect("decode must succeed");
        for k in &keys {
            assert!(
                decoded.contains(k.as_bytes()),
                "key {k} lost in bloom roundtrip (false negative)"
            );
        }
    }

    #[test]
    fn rebuild_has_low_false_positive() {
        let mut bloom = new_bloom(1000);
        for i in 0..1000 {
            bloom.insert(format!("k{i}").as_bytes());
        }
        let mut buf = Vec::new();
        encode_bloom(&mut buf, &bloom);
        let decoded = decode_bloom(&buf).expect("decode must succeed");

        let fp = (0..20_000u32)
            .filter(|i| decoded.contains(format!("absent-{i}").as_bytes()))
            .count();
        assert!(
            (fp as f64) < 0.03 * 20_000.0,
            "false positive rate too high: {fp}/20000"
        );
    }

    #[test]
    fn different_seed_breaks_rebuild() {
        let keys: Vec<String> = (0..1000).map(|i| format!("k{i}")).collect();
        let mut bloom = BloomFilter::with_false_pos(BLOOM_FP_RATE)
            .seed(&(BLOOM_SEED + 1))
            .expected_items(1000);
        for k in &keys {
            bloom.insert(k.as_bytes());
        }
        let mut buf = Vec::new();
        encode_bloom(&mut buf, &bloom);
        let decoded = decode_bloom(&buf).expect("decode must succeed");

        let still_contained = keys
            .iter()
            .filter(|k| decoded.contains(k.as_bytes()))
            .count();
        assert!(
            still_contained < 500,
            "wrong-seed rebuild must drop most present keys, {still_contained}/1000 still contained"
        );
    }

    #[test]
    fn expected_items_zero_is_valid() {
        let bloom = new_bloom(0);
        let mut buf = Vec::new();
        encode_bloom(&mut buf, &bloom);
        assert!(buf.len() >= 4);
        let decoded = decode_bloom(&buf).expect("zero-item bloom must decode");
        assert!(!decoded.contains(b"anything"));
    }

    #[test]
    fn decode_truncated_errors() {
        assert!(matches!(
            decode_bloom(&[0u8; 3]),
            Err(LsmError::Corruption(_))
        ));
        assert!(matches!(
            decode_bloom(&[0u8; 5]),
            Err(LsmError::Corruption(_))
        ));
    }

    #[test]
    fn decode_zero_words_errors() {
        // 4 bytes = num_hashes only, no words: fastbloom's from_vec panics on an
        // empty vec, so decode_bloom must reject it instead.
        let mut buf = Vec::new();
        utils::put_u32(&mut buf, 7);
        let err = decode_bloom(&buf).expect_err("zero-word block must not panic");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn decode_implausible_num_hashes_errors() {
        let mut bloom = new_bloom(100);
        bloom.insert(b"k");
        let mut buf = Vec::new();
        encode_bloom(&mut buf, &bloom);
        // Overwrite num_hashes with a hostile value.
        utils::put_u32(&mut buf, u32::MAX);
        let err = decode_bloom(&buf).expect_err("implausible num_hashes must error");
        assert!(matches!(err, LsmError::Corruption(_)));
    }

    #[test]
    fn encode_layout() {
        let bloom = new_bloom(100);
        let mut buf = Vec::new();
        encode_bloom(&mut buf, &bloom);
        assert_eq!(buf.len(), 4 + 8 * bloom.iter().count());
        assert_eq!(
            utils::read_u32(&buf, 0).expect("read num_hashes must succeed"),
            bloom.num_hashes()
        );
    }
}
