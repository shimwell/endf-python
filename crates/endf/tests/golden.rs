//! Hold the Rust reader to what the Python reader produces.
//!
//! Every file in `tests/golden/` is a reference dump written by
//! `tools/dump_golden.py`. This test finds each one, reads the ENDF file it
//! names, and compares. Adding coverage — another nuclide, another library,
//! another sublibrary — is dropping the evaluation into `tests/data/` and
//! regenerating; nothing here needs to change.
//!
//! Floats are compared exactly. The golden file records them as the shortest
//! round-tripping decimal, and both readers parse decimals with correct
//! rounding, so any difference at all is a real difference.

use std::path::{Path, PathBuf};

use endf::{materials_from_str, Material, Section};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/endf.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

/// A `key a b c` line from the golden file.
struct Record<'a> {
    key: &'a str,
    fields: Vec<&'a str>,
    line_no: usize,
}

fn records(text: &str) -> Vec<Record<'_>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .map(|(i, l)| {
            let mut parts = l.split_whitespace();
            let key = parts.next().unwrap_or("");
            Record {
                key,
                fields: parts.collect(),
                line_no: i + 1,
            }
        })
        .collect()
}

fn f64s(fields: &[&str], what: &str) -> Vec<f64> {
    fields
        .iter()
        .map(|s| {
            s.parse::<f64>()
                .unwrap_or_else(|_| panic!("{what}: {s:?} is not a float"))
        })
        .collect()
}

fn i32s(fields: &[&str], what: &str) -> Vec<i32> {
    fields
        .iter()
        .map(|s| {
            s.parse::<i32>()
                .unwrap_or_else(|_| panic!("{what}: {s:?} is not an integer"))
        })
        .collect()
}

/// Compare one golden file against the Rust reader.
fn check(golden_path: &Path) -> usize {
    let golden = std::fs::read_to_string(golden_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", golden_path.display()));
    let name = golden_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let recs = records(&golden);

    let source = recs
        .iter()
        .find(|r| r.key == "SOURCE")
        .unwrap_or_else(|| panic!("{name} has no SOURCE record"))
        .fields[0];
    let source_path = repo_root().join(source);
    let text = std::fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", source_path.display()));

    let materials = materials_from_str(&text)
        .unwrap_or_else(|e| panic!("{name}: the Rust reader failed on {source}: {e}"));

    let mut material: Option<&Material> = None;
    let mut mat_index = 0usize;
    // Carried between the MF3 header record and the tables that follow it.
    let mut pending_mf3: Option<(i32, usize, usize)> = None;
    // The abscissae of the most recent EVALX, awaiting their EVALY.
    let mut pending_eval: Vec<f64> = Vec::new();
    let mut checks = 0usize;

    for rec in &recs {
        let at = format!("{name}:{}", rec.line_no);
        match rec.key {
            "SOURCE" => {}
            "MATERIALS" => {
                let n: usize = rec.fields[0].parse().unwrap();
                assert_eq!(materials.len(), n, "{at}: material count");
                checks += 1;
            }
            "MAT" => {
                let want: i32 = rec.fields[0].parse().unwrap();
                let m = materials
                    .get(mat_index)
                    .unwrap_or_else(|| panic!("{at}: no material {mat_index}"));
                assert_eq!(m.mat, want, "{at}: material number");
                material = Some(m);
                mat_index += 1;
                checks += 1;
            }
            "SECTION" => {
                let m = material.expect("SECTION before MAT");
                let (mf, mt) = (
                    rec.fields[0].parse().unwrap(),
                    rec.fields[1].parse().unwrap(),
                );
                let want_lines: usize = rec.fields[2].parse().unwrap();
                let body = m
                    .section_text
                    .get(&(mf, mt))
                    .unwrap_or_else(|| panic!("{at}: MF={mf} MT={mt} is missing"));
                assert_eq!(
                    body.lines().count(),
                    want_lines,
                    "{at}: MF={mf} MT={mt} line count"
                );
                checks += 1;
            }
            "MF3" => {
                let m = material.expect("MF3 before MAT");
                let mt: i32 = rec.fields[0].parse().unwrap();
                let got = m
                    .mf3(mt)
                    .unwrap_or_else(|| panic!("{at}: MF=3 MT={mt} did not parse"));

                assert_eq!(got.za, rec.fields[1].parse::<i64>().unwrap(), "{at}: ZA");
                assert_eq!(got.awr, rec.fields[2].parse::<f64>().unwrap(), "{at}: AWR");
                assert_eq!(got.qm, rec.fields[3].parse::<f64>().unwrap(), "{at}: QM");
                assert_eq!(got.qi, rec.fields[4].parse::<f64>().unwrap(), "{at}: QI");
                assert_eq!(got.lr, rec.fields[5].parse::<i64>().unwrap(), "{at}: LR");

                let n_pairs: usize = rec.fields[6].parse().unwrap();
                let n_regions: usize = rec.fields[7].parse().unwrap();
                assert_eq!(
                    got.sigma.n_pairs(),
                    n_pairs,
                    "{at}: MT={mt} number of pairs"
                );
                assert_eq!(
                    got.sigma.n_regions(),
                    n_regions,
                    "{at}: MT={mt} number of regions"
                );
                pending_mf3 = Some((mt, n_pairs, n_regions));
                checks += 1;
            }
            "BP" | "INT" | "X" | "Y" => {
                let m = material.expect("a table before MAT");
                let (mt, n_pairs, n_regions) = pending_mf3.expect("a table before MF3");
                let got = m.mf3(mt).unwrap();
                match rec.key {
                    "BP" => {
                        let want = i32s(&rec.fields, &at);
                        assert_eq!(want.len(), n_regions, "{at}: breakpoint count");
                        assert_eq!(got.sigma.breakpoints, want, "{at}: MT={mt} breakpoints");
                    }
                    "INT" => {
                        let want = i32s(&rec.fields, &at);
                        assert_eq!(got.sigma.interpolation, want, "{at}: MT={mt} interpolation");
                    }
                    "X" => {
                        let want = f64s(&rec.fields, &at);
                        assert_eq!(want.len(), n_pairs, "{at}: x count");
                        assert_eq!(got.sigma.x, want, "{at}: MT={mt} x values");
                    }
                    "Y" => {
                        let want = f64s(&rec.fields, &at);
                        assert_eq!(got.sigma.y, want, "{at}: MT={mt} y values");
                    }
                    _ => unreachable!(),
                }
                checks += 1;
            }
            "EVALX" => {
                pending_eval = f64s(&rec.fields, &at);
            }
            "EVALY" => {
                let m = material.expect("EVALY before MAT");
                let (mt, ..) = pending_mf3.expect("EVALY before MF3");
                let want = f64s(&rec.fields, &at);
                assert_eq!(want.len(), pending_eval.len(), "{at}: EVALX/EVALY length");
                let got: Vec<f64> = pending_eval
                    .iter()
                    .map(|&x| m.mf3(mt).unwrap().sigma.eval(x))
                    .collect();
                // Interpolation is arithmetic, not parsing: the two languages
                // evaluate the same expression but need not round identically
                // once logs and exps are involved, so this is the one
                // comparison held to a tolerance rather than to equality.
                for ((&x, &w), &g) in pending_eval.iter().zip(&want).zip(&got) {
                    let tol = 1e-12 * w.abs().max(1.0);
                    assert!(
                        (g - w).abs() <= tol,
                        "{at}: MT={mt} at x={x}: rust {g} != python {w}"
                    );
                }
                checks += 1;
            }
            other => panic!("{at}: unknown golden record {other:?}"),
        }
    }

    // The golden file lists every section, so anything extra on the Rust side
    // is a splitting bug the record-by-record walk above cannot see.
    let listed: usize = recs.iter().filter(|r| r.key == "SECTION").count();
    let parsed: usize = materials.iter().map(|m| m.section_text.len()).sum();
    assert_eq!(
        parsed, listed,
        "{name}: the Rust reader found sections the golden file does not list"
    );

    checks
}

#[test]
fn matches_the_python_reader() {
    let dir = golden_dir();
    let mut goldens: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    goldens.sort();

    assert!(!goldens.is_empty(), "no golden files in {}", dir.display());

    let mut total = 0usize;
    for path in &goldens {
        total += check(path);
    }
    println!("{} golden files, {total} assertions", goldens.len());
}

#[test]
fn unported_files_keep_their_text() {
    // The port proceeds file by file, so a section with no Rust parser must
    // still round-trip its text for the Python reader to fall back to.
    let path = repo_root().join("tests").join("n-095_Am_244.endf");
    let text = std::fs::read_to_string(&path).unwrap();
    let m = Material::from_str(&text).unwrap();

    let unparsed: Vec<(i32, i32)> = m
        .section_data
        .iter()
        .filter(|(_, s)| matches!(s, Section::Unparsed { .. }))
        .map(|(&k, _)| k)
        .collect();
    assert!(
        !unparsed.is_empty(),
        "expected files that are not ported yet"
    );
    for key in unparsed {
        let body = &m.section_text[&key];
        assert!(!body.is_empty(), "MF={} MT={} lost its text", key.0, key.1);
    }
}
