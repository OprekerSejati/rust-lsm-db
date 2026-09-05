use thiserror::Error;

#[derive(Error, Debug)]
pub enum LsmError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupted data: {0}")]
    Corruption(String),
    #[error("key length {0} exceeds maximum {1}")]
    KeyTooLarge(usize, usize),
    #[error("value length {0} exceeds maximum {1}")]
    ValueTooLarge(usize, usize),
    #[error("invalid operation: {0}")]
    InvalidOp(String),
}

pub type Result<T> = std::result::Result<T, LsmError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn force_io_error() -> Result<()> {
        let io_result: std::io::Result<()> = Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "simulated io failure",
        ));
        Ok(io_result?)
    }

    #[test]
    fn question_mark_converts_io_error_to_lsm_error() {
        let err = force_io_error().expect_err("force_io_error should return an error");
        assert!(
            matches!(err, LsmError::Io(_)),
            "expected LsmError::Io variant, got: {err}"
        );
    }

    #[test]
    fn display_format_matches_expected_messages() {
        let io = LsmError::Io(std::io::Error::other("disk read failed"));
        assert_eq!(io.to_string(), "I/O error: disk read failed");

        let corruption = LsmError::Corruption("checksum mismatch".to_string());
        assert_eq!(corruption.to_string(), "corrupted data: checksum mismatch");

        let key = LsmError::KeyTooLarge(1024, 64);
        assert_eq!(key.to_string(), "key length 1024 exceeds maximum 64");

        let value = LsmError::ValueTooLarge(65_536, 4096);
        assert_eq!(value.to_string(), "value length 65536 exceeds maximum 4096");

        let invalid = LsmError::InvalidOp("delete on read-only engine".to_string());
        assert_eq!(
            invalid.to_string(),
            "invalid operation: delete on read-only engine"
        );
    }

    #[test]
    fn display_output_is_non_empty_for_all_variants() {
        let messages = [
            LsmError::Io(std::io::Error::other("x")).to_string(),
            LsmError::Corruption("x".to_string()).to_string(),
            LsmError::KeyTooLarge(1, 2).to_string(),
            LsmError::ValueTooLarge(1, 2).to_string(),
            LsmError::InvalidOp("x".to_string()).to_string(),
        ];
        for msg in messages {
            assert!(!msg.is_empty(), "display output must not be empty");
        }
    }
}
