//! Little-endian primitive (de)serialization over byte slices/buffers.
//! Reads are bounds-checked and surface `LsmError::Corruption` on short input.

use crate::error::{LsmError, Result};

pub fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        read_exact(bytes, offset, 2)?.try_into().unwrap(),
    ))
}

pub fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        read_exact(bytes, offset, 4)?.try_into().unwrap(),
    ))
}

pub fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        read_exact(bytes, offset, 8)?.try_into().unwrap(),
    ))
}

// Bounds check first; the length-homogeneous slice conversion below is then
// infallible, so the unwraps in the public readers are safe by construction.
fn read_exact(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    let end = offset.checked_add(size).ok_or_else(|| {
        LsmError::Corruption(format!(
            "read of {size} bytes overflows usize at offset {offset}"
        ))
    })?;
    if end > bytes.len() {
        return Err(LsmError::Corruption(format!(
            "read of {size} bytes at offset {offset} exceeds buffer length {}",
            bytes.len()
        )));
    }
    Ok(&bytes[offset..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LsmError;

    #[test]
    fn put_read_roundtrip_u16() {
        let mut buf = Vec::new();
        put_u16(&mut buf, 0x0102);
        assert_eq!(
            read_u16(&buf, 0).expect("in-bounds read must succeed"),
            0x0102
        );
    }

    #[test]
    fn put_read_roundtrip_u32() {
        let mut buf = Vec::new();
        put_u32(&mut buf, 0xDEAD_BEEF);
        assert_eq!(
            read_u32(&buf, 0).expect("in-bounds read must succeed"),
            0xDEAD_BEEF
        );
    }

    #[test]
    fn put_read_roundtrip_u64() {
        let mut buf = Vec::new();
        put_u64(&mut buf, 0x0123_4567_89AB_CDEF);
        assert_eq!(
            read_u64(&buf, 0).expect("in-bounds read must succeed"),
            0x0123_4567_89AB_CDEF
        );
    }

    #[test]
    fn little_endian_byte_order() {
        let mut buf = Vec::new();
        put_u16(&mut buf, 0x0102);
        assert_eq!(buf, [0x02, 0x01]);

        let mut buf = Vec::new();
        put_u32(&mut buf, 0x0102_0304);
        assert_eq!(buf, [0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn reads_at_offset_into_larger_buffer() {
        let mut buf = Vec::new();
        put_u32(&mut buf, 1);
        put_u32(&mut buf, 2);
        assert_eq!(read_u32(&buf, 4).expect("in-bounds read must succeed"), 2);
    }

    #[test]
    fn read_out_of_bounds_errors() {
        assert!(matches!(read_u16(&[], 0), Err(LsmError::Corruption(_))));
        assert!(matches!(
            read_u32(&[0u8; 3], 0),
            Err(LsmError::Corruption(_))
        ));
        assert!(matches!(
            read_u64(&[0u8; 7], 0),
            Err(LsmError::Corruption(_))
        ));
        assert!(matches!(
            read_u16(&[0u8; 4], 3),
            Err(LsmError::Corruption(_))
        ));
    }

    #[test]
    fn read_exactly_at_end_ok() {
        assert_eq!(
            read_u16(&[0, 1], 0).expect("read exactly at buffer end must succeed"),
            0x0100
        );
    }

    #[test]
    fn mixed_width_sequential_layout() {
        let mut buf = Vec::new();
        put_u16(&mut buf, 0xABCD);
        put_u32(&mut buf, 0x1122_3344);
        put_u64(&mut buf, 0x0102_0304_0506_0708);
        assert_eq!(buf.len(), 2 + 4 + 8);

        assert_eq!(read_u16(&buf, 0).expect("read u16"), 0xABCD);
        assert_eq!(read_u32(&buf, 2).expect("read u32"), 0x1122_3344);
        assert_eq!(read_u64(&buf, 6).expect("read u64"), 0x0102_0304_0506_0708);
    }

    #[test]
    fn read_at_offset_equal_to_len_errors() {
        assert!(matches!(read_u16(&[0, 1], 2), Err(LsmError::Corruption(_))));
    }
}
