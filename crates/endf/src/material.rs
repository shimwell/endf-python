//! Splitting an ENDF-6 file into materials and their (MF, MT) sections.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::mf;
use crate::records::{field, Reader};

/// A parsed section, one variant per MF/MT family.
///
/// [`Section::Unparsed`] covers the files not yet ported to Rust. Their text is
/// still kept in [`Material::section_text`], so a consumer can fall back to the
/// Python reader for them while the port proceeds file by file.
#[derive(Debug, Clone, PartialEq)]
pub enum Section {
    /// MF=3, reaction cross sections.
    Mf3(mf::mf3::Mf3),
    Unparsed {
        mf: i32,
        mt: i32,
    },
}

/// An ENDF material: one evaluation, made up of (MF, MT) sections.
#[derive(Debug, Clone)]
pub struct Material {
    /// ENDF material number.
    pub mat: i32,
    /// Raw text of each section, keyed by (MF, MT), control columns included.
    pub section_text: BTreeMap<(i32, i32), String>,
    /// Parsed form of each section.
    pub section_data: BTreeMap<(i32, i32), Section>,
}

/// Read a MAT/MF/MT control field.
///
/// A blank field reads as zero, which is what the format means by it. Anything
/// present but not an integer is a malformed line and an error, rather than
/// being silently taken as zero.
fn control_field(line: &str, start: usize, end: usize) -> Result<i32> {
    let s = field(line, start, end).trim();
    if s.is_empty() {
        return Ok(0);
    }
    s.parse::<i32>().map_err(|_| Error::BadControlField {
        line: line.to_string(),
    })
}

/// The (MAT, MF, MT) control fields in the last 14 columns of a line.
fn control(line: &str) -> Result<(i32, i32, i32)> {
    Ok((
        control_field(line, 66, 70)?,
        control_field(line, 70, 72)?,
        control_field(line, 72, 75)?,
    ))
}

impl Material {
    /// Parse the first material in `text`, skipping the leading TPID record.
    ///
    /// Not `std::str::FromStr`: that trait's `Err` would have to be this
    /// crate's error type anyway, and the inherent method keeps the name the
    /// Python reader uses.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Result<Material> {
        let lines: Vec<&str> = text.lines().collect();
        let mut cursor = usize::from(!lines.is_empty());
        Material::parse(&lines, &mut cursor)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Material> {
        let text = std::fs::read_to_string(path)?;
        Material::from_str(&text)
    }

    /// Parse one material starting at `cursor`, leaving `cursor` on the line
    /// after the material's MEND record.
    fn parse(lines: &[&str], cursor: &mut usize) -> Result<Material> {
        // Evaluators sometimes write ill-formed TPID records, so the material
        // number is taken from the first line that actually belongs to a file
        // (MF != 0) rather than from the header.
        let mat = loop {
            let line = lines.get(*cursor).ok_or(Error::UnexpectedEof {
                expected: "the start of a material",
            })?;
            let (m, mf, _) = control(line)?;
            if mf != 0 {
                break m;
            }
            *cursor += 1;
        };

        let mut section_text: BTreeMap<(i32, i32), String> = BTreeMap::new();
        loop {
            // Advance to the next section head, or to the end of the material.
            let (cur_mat, mf, mt) = loop {
                let line = lines.get(*cursor).ok_or(Error::UnexpectedEof {
                    expected: "the next section",
                })?;
                let c = control(line)?;
                if c.2 > 0 || c.0 == 0 {
                    break c;
                }
                *cursor += 1;
            };

            // MAT=0 is the MEND record: this material is done.
            if cur_mat == 0 {
                *cursor += 1;
                break;
            }

            // Collect the section's lines up to its SEND record (MT=0).
            let mut body = String::new();
            loop {
                let line = lines.get(*cursor).ok_or(Error::UnexpectedEof {
                    expected: "the end of a section",
                })?;
                *cursor += 1;
                if control(line)?.2 == 0 {
                    break;
                }
                body.push_str(line);
                body.push('\n');
            }
            section_text.insert((mf, mt), body);
        }

        let mut section_data = BTreeMap::new();
        for (&(mf, mt), text) in &section_text {
            section_data.insert((mf, mt), parse_section(mf, mt, text)?);
        }

        Ok(Material {
            mat,
            section_text,
            section_data,
        })
    }

    /// The (MF, MT) sections present, in ascending order.
    pub fn sections(&self) -> Vec<(i32, i32)> {
        self.section_text.keys().copied().collect()
    }

    pub fn contains(&self, mf: i32, mt: i32) -> bool {
        self.section_data.contains_key(&(mf, mt))
    }

    pub fn get(&self, mf: i32, mt: i32) -> Option<&Section> {
        self.section_data.get(&(mf, mt))
    }

    /// The MF=3 cross section for a reaction, if the material has one.
    pub fn mf3(&self, mt: i32) -> Option<&mf::mf3::Mf3> {
        match self.section_data.get(&(3, mt))? {
            Section::Mf3(s) => Some(s),
            _ => None,
        }
    }
}

/// Dispatch a section to its file's parser.
///
/// Each MF ported to Rust adds an arm here; everything else stays
/// [`Section::Unparsed`] with its text preserved.
fn parse_section(mf: i32, mt: i32, text: &str) -> Result<Section> {
    let mut r = Reader::new(text);
    Ok(match mf {
        3 => Section::Mf3(mf::mf3::parse_mf3(&mut r)?),
        _ => Section::Unparsed { mf, mt },
    })
}

/// Read every material in an ENDF-6 file.
pub fn get_materials(path: impl AsRef<Path>) -> Result<Vec<Material>> {
    let text = std::fs::read_to_string(path)?;
    materials_from_str(&text)
}

/// Read every material in the text of an ENDF-6 file.
pub fn materials_from_str(text: &str) -> Result<Vec<Material>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(Vec::new());
    }
    // Skip the TPID record that opens the file.
    let mut cursor = 1usize;
    let mut materials = Vec::new();
    while cursor < lines.len() {
        // MAT=-1 is the TEND record that closes the file.
        if control_field(lines[cursor], 66, 70)? == -1 {
            break;
        }
        let before = cursor;
        materials.push(Material::parse(&lines, &mut cursor)?);
        debug_assert!(cursor > before, "parsing a material must consume lines");
    }
    Ok(materials)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../tests/n-095_Am_244.endf");

    #[test]
    fn reads_the_material_number() {
        let m = Material::from_str(FIXTURE).unwrap();
        assert_eq!(m.mat, 9552);
    }

    #[test]
    fn finds_the_expected_sections() {
        let m = Material::from_str(FIXTURE).unwrap();
        let sections = m.sections();
        // The evaluation opens with the descriptive MF=1/MT=451 section and
        // carries resonance parameters and cross sections.
        assert!(sections.contains(&(1, 451)));
        assert!(sections.contains(&(2, 151)));
        assert!(sections.contains(&(3, 1)));
        assert!(sections.contains(&(3, 2)));
        assert!(sections.contains(&(3, 18)));
        assert!(!sections.is_empty());
    }

    #[test]
    fn section_text_excludes_the_send_record() {
        let m = Material::from_str(FIXTURE).unwrap();
        let text = &m.section_text[&(3, 1)];
        for line in text.lines() {
            let (_, mf, mt) = control(line).unwrap();
            assert_eq!(mf, 3);
            assert_ne!(mt, 0, "a SEND record leaked into the section body");
        }
    }

    #[test]
    fn every_section_is_dispatched() {
        let m = Material::from_str(FIXTURE).unwrap();
        assert_eq!(m.section_text.len(), m.section_data.len());
    }

    #[test]
    fn reads_the_whole_file_as_one_material() {
        let materials = materials_from_str(FIXTURE).unwrap();
        assert_eq!(materials.len(), 1);
        assert_eq!(materials[0].mat, 9552);
    }
}
