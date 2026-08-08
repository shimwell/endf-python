//! Errors raised while reading an ENDF-6 file.

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// A record was expected but the section ended first.
    UnexpectedEof {
        /// What the reader was trying to read.
        expected: &'static str,
    },
    /// A control field (MAT/MF/MT) could not be read as an integer.
    BadControlField {
        line: String,
    },
    /// An INTG record used an NDIGIT value the format does not define.
    BadNdigit {
        ndigit: i64,
    },
    /// A representation the reader does not implement, matching where the
    /// Python reader raises `NotImplementedError`.
    Unsupported {
        what: &'static str,
    },
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnexpectedEof { expected } => {
                write!(f, "ENDF section ended while reading {expected}")
            }
            Error::BadControlField { line } => {
                write!(
                    f,
                    "could not read MAT/MF/MT control fields from line: {line:?}"
                )
            }
            Error::BadNdigit { ndigit } => {
                write!(f, "INTG record has NDIGIT={ndigit}, expected 2 through 6")
            }
            Error::Unsupported { what } => {
                write!(f, "this reader does not implement {what}")
            }
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
