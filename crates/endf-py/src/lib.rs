//! Python bindings for the `endf` crate.
//!
//! Deliberately thin: every type here forwards to the Rust one and converts at
//! the boundary. Interpretation belongs in the Rust crate so that consumers
//! which never load Python get the same behaviour.
//!
//! The concrete types — a tabulated function, a reaction, a nuclide — are
//! wrapped as classes. The sum types — an angle-energy distribution, an
//! outgoing energy law, a univariate density — come across as dicts tagged
//! with a `kind` key. That is the shape a consumer wants anyway: `kind` is
//! exactly the discriminant an Arrow union column needs, and it saves a
//! wrapper class per variant for no gain in what can be expressed.
//!
//! This is not a drop-in for `endf.Material.section_data`, which returns the
//! Python reader's own dictionaries keyed by ENDF field name. What is exposed
//! here is the typed layer above that.

use std::collections::BTreeMap;

use endf::angle_energy::AngleEnergy;
use endf::mf::mf4::{AngleAtEnergy, AngleDistribution};
use endf::mf::mf5::EnergyDistribution;
use endf::univariate::Univariate;
use endf::Section;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py_err(e: endf::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Read a file, decompressing it when the name says it is compressed.
///
/// The Python package reads `.xz` through `endf.fileutils.open_text`; a path
/// that works there has to work here.
fn read_text(filename: &str) -> PyResult<String> {
    let raw = std::fs::read(filename)
        .map_err(|e| PyValueError::new_err(format!("reading {filename}: {e}")))?;
    if !filename.ends_with(".xz") {
        return String::from_utf8(raw)
            .map_err(|e| PyValueError::new_err(format!("{filename} is not UTF-8: {e}")));
    }
    let mut out = Vec::new();
    lzma_rs::xz_decompress(&mut raw.as_slice(), &mut out)
        .map_err(|e| PyValueError::new_err(format!("decompressing {filename}: {e}")))?;
    String::from_utf8(out)
        .map_err(|e| PyValueError::new_err(format!("{filename} is not UTF-8: {e}")))
}

/// Convert a string from an ENDF floating point field to a float.
#[pyfunction]
#[pyo3(name = "float_endf")]
fn py_float_endf(s: &str) -> f64 {
    endf::float_endf(s)
}

/// Convert a string from an ENDF integer field to an int.
#[pyfunction]
#[pyo3(name = "int_endf")]
fn py_int_endf(s: &str) -> i64 {
    endf::int_endf(s)
}

/// A one-dimensional tabulated function (the format's TAB1 type).
#[pyclass(name = "Tabulated1D", module = "endf._endf")]
#[derive(Clone)]
struct PyTabulated1D {
    inner: endf::Tabulated1D,
}

#[pymethods]
impl PyTabulated1D {
    #[new]
    #[pyo3(signature = (x, y, breakpoints=None, interpolation=None))]
    fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        breakpoints: Option<Vec<i32>>,
        interpolation: Option<Vec<i32>>,
    ) -> Self {
        let inner = match (breakpoints, interpolation) {
            (Some(b), Some(i)) => endf::Tabulated1D::with_regions(x, y, b, i),
            _ => endf::Tabulated1D::new(x, y),
        };
        PyTabulated1D { inner }
    }

    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    #[getter]
    fn y(&self) -> Vec<f64> {
        self.inner.y.clone()
    }

    #[getter]
    fn breakpoints(&self) -> Vec<i32> {
        self.inner.breakpoints.clone()
    }

    #[getter]
    fn interpolation(&self) -> Vec<i32> {
        self.inner.interpolation.clone()
    }

    #[getter]
    fn n_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    #[getter]
    fn n_regions(&self) -> usize {
        self.inner.n_regions()
    }

    /// Evaluate at a point, or elementwise over a sequence.
    fn __call__<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // A sequence first: a float extracts from a 1-element sequence in some
        // cases, but a sequence never extracts as a float.
        if let Ok(xs) = x.extract::<Vec<f64>>() {
            let ys: Vec<f64> = xs.iter().map(|&v| self.inner.eval(v)).collect();
            return Ok(ys.into_pyobject(py)?.into_any());
        }
        let v: f64 = x.extract()?;
        Ok(self.inner.eval(v).into_pyobject(py)?.into_any())
    }

    /// Partial integrals from the start of the range to each tabulated point.
    fn integral(&self) -> Vec<f64> {
        self.inner.integral()
    }

    fn __len__(&self) -> usize {
        self.inner.n_pairs()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Tabulated1D: {} points, {} regions>",
            self.inner.n_pairs(),
            self.inner.n_regions()
        )
    }
}

/// Interpolation metadata for a two-dimensional function (the format's TAB2).
#[pyclass(name = "Tabulated2D", module = "endf._endf")]
#[derive(Clone)]
struct PyTabulated2D {
    inner: endf::Tabulated2D,
}

#[pymethods]
impl PyTabulated2D {
    #[getter]
    fn breakpoints(&self) -> Vec<i32> {
        self.inner.breakpoints.clone()
    }

    #[getter]
    fn interpolation(&self) -> Vec<i32> {
        self.inner.interpolation.clone()
    }

    fn __repr__(&self) -> String {
        format!("<Tabulated2D: {} regions>", self.inner.breakpoints.len())
    }
}

/// An MF=3 reaction cross section.
#[pyclass(name = "CrossSection", module = "endf._endf")]
#[derive(Clone)]
struct PyCrossSection {
    inner: endf::mf::mf3::Mf3,
}

#[pymethods]
impl PyCrossSection {
    #[getter]
    #[allow(non_snake_case)]
    fn ZA(&self) -> i64 {
        self.inner.za
    }

    #[getter]
    #[allow(non_snake_case)]
    fn AWR(&self) -> f64 {
        self.inner.awr
    }

    #[getter]
    #[allow(non_snake_case)]
    fn QM(&self) -> f64 {
        self.inner.qm
    }

    #[getter]
    #[allow(non_snake_case)]
    fn QI(&self) -> f64 {
        self.inner.qi
    }

    #[getter]
    #[allow(non_snake_case)]
    fn LR(&self) -> i64 {
        self.inner.lr
    }

    #[getter]
    fn sigma(&self) -> PyTabulated1D {
        PyTabulated1D {
            inner: self.inner.sigma.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<CrossSection: ZA={}, {} points>",
            self.inner.za,
            self.inner.sigma.n_pairs()
        )
    }
}

/// An ENDF material with multiple files/sections.
#[pyclass(name = "Material", module = "endf._endf")]
#[derive(Clone)]
struct PyMaterial {
    inner: endf::Material,
}

#[pymethods]
impl PyMaterial {
    #[new]
    fn new(filename: &str) -> PyResult<Self> {
        let inner = endf::Material::from_str(&read_text(filename)?).map_err(to_py_err)?;
        Ok(PyMaterial { inner })
    }

    /// Parse a material from the text of an ENDF file.
    #[staticmethod]
    fn from_string(text: &str) -> PyResult<Self> {
        let inner = endf::Material::from_str(text).map_err(to_py_err)?;
        Ok(PyMaterial { inner })
    }

    #[getter]
    #[allow(non_snake_case)]
    fn MAT(&self) -> i32 {
        self.inner.mat
    }

    /// The (MF, MT) sections present.
    #[getter]
    fn sections(&self) -> Vec<(i32, i32)> {
        self.inner.sections()
    }

    /// Raw text of each section, keyed by (MF, MT).
    #[getter]
    fn section_text(&self) -> BTreeMap<(i32, i32), String> {
        self.inner.section_text.clone()
    }

    /// Every section as the Python reader's dictionaries, keyed by (MF, MT).
    ///
    /// Sections with no dictionary form here are left out rather than
    /// half-built; they are reached through the object layer, which covers
    /// every file.
    #[getter]
    fn section_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (&key, section) in &self.inner.section_data {
            if let Some(d) = section_dict(py, section)? {
                out.set_item(key, d)?;
            }
        }
        Ok(out)
    }

    /// One section's dictionary, as `material[3, 1]` gives it.
    fn __getitem__<'py>(&self, py: Python<'py>, key: (i32, i32)) -> PyResult<Bound<'py, PyDict>> {
        let (mf, mt) = key;
        let section = self
            .inner
            .section_data
            .get(&key)
            .ok_or_else(|| PyValueError::new_err(format!("no section MF={mf} MT={mt}")))?;
        section_dict(py, section)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "MF={mf} MT={mt} has no dictionary form here; the typed \
                 accessors and the object layer reach it instead"
            ))
        })
    }

    /// The MF=3 cross section for a reaction, or None.
    fn mf3(&self, mt: i32) -> Option<PyCrossSection> {
        self.inner
            .mf3(mt)
            .map(|s| PyCrossSection { inner: s.clone() })
    }

    fn __contains__(&self, key: (i32, i32)) -> bool {
        self.inner.contains(key.0, key.1)
    }

    fn __repr__(&self) -> String {
        format!(
            "<Material: MAT={}, {} sections>",
            self.inner.mat,
            self.inner.section_text.len()
        )
    }
}

// ---------------------------------------------------------------------------
// section_data
//
// The Python reader hands back a dictionary per section, keyed by the field
// names the format uses. This rebuilds those dictionaries from the typed
// sections, so code written against `Material.section_data` keeps working.
//
// Not every file has a projection yet — `section_dict` says which by name
// rather than returning something incomplete. The typed accessors and the
// object layer above them cover every file regardless.
// ---------------------------------------------------------------------------

fn tab1_class(t: &endf::Tabulated1D) -> PyTabulated1D {
    PyTabulated1D { inner: t.clone() }
}

fn tab2_class(t: &endf::Tabulated2D) -> PyTabulated2D {
    PyTabulated2D { inner: t.clone() }
}

fn nu_into(d: &Bound<'_, PyDict>, nu: &endf::mf::mf1::Nu) -> PyResult<()> {
    match nu {
        endf::mf::mf1::Nu::Polynomial(c) => d.set_item("C", c.clone())?,
        endf::mf::mf1::Nu::Tabulated(t) => d.set_item("nu", tab1_class(t))?,
        endf::mf::mf1::Nu::Absent => {}
    }
    Ok(())
}

fn mf1_mt451_dict<'py>(
    py: Python<'py>,
    s: &endf::mf::mf1::Mf1Mt451,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (key, value) in [
        ("ZA", s.za),
        ("LRP", s.lrp),
        ("LFI", s.lfi),
        ("NLIB", s.nlib),
        ("NMOD", s.nmod),
        ("LIS", s.lis),
        ("LISO", s.liso),
        ("NFOR", s.nfor),
        ("LREL", s.lrel),
        ("NSUB", s.nsub),
        ("NVER", s.nver),
        ("LDRV", s.ldrv),
        ("NWD", s.nwd),
        ("NXC", s.nxc),
    ] {
        d.set_item(key, value)?;
    }
    for (key, value) in [
        ("AWR", s.awr),
        ("ELIS", s.elis),
        ("STA", s.sta),
        ("AWI", s.awi),
        ("EMAX", s.emax),
        ("TEMP", s.temp),
    ] {
        d.set_item(key, value)?;
    }
    // The descriptive text is present only when the evaluation wrote it.
    if let Some(zsymam) = &s.zsymam {
        d.set_item("ZSYMAM", zsymam)?;
        d.set_item("ALAB", &s.alab)?;
        d.set_item("EDATE", &s.edate)?;
        d.set_item("AUTH", &s.auth)?;
        d.set_item("REF", &s.reference)?;
        d.set_item("DDATE", &s.ddate)?;
        d.set_item("RDATE", &s.rdate)?;
        d.set_item("ENDATE", &s.endate)?;
        d.set_item("HSUB", s.hsub.clone())?;
        d.set_item("description", s.description.clone())?;
    }
    d.set_item("section_list", s.section_list.clone())?;
    Ok(d)
}

fn mf3_dict<'py>(py: Python<'py>, s: &endf::mf::mf3::Mf3) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("QM", s.qm)?;
    d.set_item("QI", s.qi)?;
    d.set_item("LR", s.lr)?;
    d.set_item("sigma", tab1_class(&s.sigma))?;
    Ok(d)
}

fn mf4_dict<'py>(py: Python<'py>, s: &endf::mf::mf4::Mf4) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LTT", s.ltt)?;
    d.set_item("LI", s.li)?;
    d.set_item("LCT", s.lct)?;
    if let Some(l) = &s.legendre {
        let sub = PyDict::new(py);
        sub.set_item("E_int", tab2_class(&l.e_int))?;
        sub.set_item("T", l.t)?;
        sub.set_item("LT", l.lt)?;
        sub.set_item("E", l.energy.clone())?;
        sub.set_item("a_l", l.a_l.clone())?;
        d.set_item("legendre", sub)?;
    }
    if let Some(t) = &s.tabulated {
        let sub = PyDict::new(py);
        sub.set_item("E_int", tab2_class(&t.e_int))?;
        sub.set_item("T", t.t)?;
        sub.set_item("LT", t.lt)?;
        sub.set_item("E", t.energy.clone())?;
        sub.set_item("mu", t.mu.iter().map(tab1_class).collect::<Vec<_>>())?;
        d.set_item("tabulated", sub)?;
    }
    Ok(d)
}

fn mf5_dict<'py>(py: Python<'py>, s: &endf::mf::mf5::Mf5) -> PyResult<Bound<'py, PyDict>> {
    use endf::mf::mf5::EnergyDistribution as E;
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("NK", s.nk)?;
    let mut subsections = Vec::with_capacity(s.subsections.len());
    for sub in &s.subsections {
        let entry = PyDict::new(py);
        entry.set_item("LF", sub.lf)?;
        entry.set_item("p", tab1_class(&sub.p))?;
        let dist = PyDict::new(py);
        match &sub.distribution {
            E::ArbitraryTabulated { e_int, energy, g } => {
                dist.set_item("E_int", tab2_class(e_int))?;
                dist.set_item("E", energy.clone())?;
                dist.set_item("g", g.iter().map(tab1_class).collect::<Vec<_>>())?;
            }
            E::GeneralEvaporation { u, theta, g } => {
                dist.set_item("U", *u)?;
                dist.set_item("theta", tab1_class(theta))?;
                dist.set_item("g", tab1_class(g))?;
            }
            E::MaxwellEnergy { u, theta } | E::Evaporation { u, theta } => {
                dist.set_item("U", *u)?;
                dist.set_item("theta", tab1_class(theta))?;
            }
            E::WattEnergy { u, a, b } => {
                dist.set_item("U", *u)?;
                dist.set_item("a", tab1_class(a))?;
                dist.set_item("b", tab1_class(b))?;
            }
            E::MadlandNix { efl, efh, t_m } => {
                dist.set_item("EFL", *efl)?;
                dist.set_item("EFH", *efh)?;
                dist.set_item("T_M", tab1_class(t_m))?;
            }
            // The three ACE-only laws have no ENDF section and so never reach
            // here; `dist` stays empty rather than inventing keys.
            _ => {}
        }
        entry.set_item("distribution", dist)?;
        subsections.push(entry);
    }
    d.set_item("subsections", subsections)?;
    Ok(d)
}

fn mf9_mf10_dict<'py>(py: Python<'py>, s: &endf::mf::mf8::Mf9Mf10) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LIS", s.lis)?;
    d.set_item("NS", s.ns)?;
    let key = if s.mf == 9 { "Y" } else { "sigma" };
    let levels: PyResult<Vec<_>> = s
        .levels
        .iter()
        .map(|level| {
            let e = PyDict::new(py);
            e.set_item("QM", level.qm)?;
            e.set_item("QI", level.qi)?;
            e.set_item("IZAP", level.izap)?;
            e.set_item("LFS", level.lfs)?;
            e.set_item(key, tab1_class(&level.func))?;
            Ok(e)
        })
        .collect();
    d.set_item("levels", levels?)?;
    Ok(d)
}

fn mf1_mt452_dict<'py>(
    py: Python<'py>,
    s: &endf::mf::mf1::Mf1Mt452,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LNU", s.lnu)?;
    nu_into(&d, &s.nu)?;
    Ok(d)
}

fn mf1_mt455_dict<'py>(
    py: Python<'py>,
    s: &endf::mf::mf1::Mf1Mt455,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LDG", s.ldg)?;
    d.set_item("LNU", s.lnu)?;
    if !s.lambda.is_empty() {
        d.set_item("lambda", s.lambda.clone())?;
    }
    if let Some(e_int) = &s.e_int {
        d.set_item("E_int", tab2_class(e_int))?;
    }
    if !s.constants.is_empty() {
        let constants: PyResult<Vec<_>> = s
            .constants
            .iter()
            .map(|c| {
                let e = PyDict::new(py);
                e.set_item("E", c.energy)?;
                e.set_item("lambda", c.lambda.clone())?;
                e.set_item("alpha", c.alpha.clone())?;
                Ok(e)
            })
            .collect();
        d.set_item("constants", constants?)?;
    }
    nu_into(&d, &s.nu)?;
    Ok(d)
}

fn mf12_dict<'py>(py: Python<'py>, s: &endf::mf::photon::Mf12) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LO", s.lo)?;
    d.set_item("NK", s.nk)?;
    if let Some(y) = &s.total_yield {
        d.set_item("Y", tab1_class(y))?;
    }
    if !s.multiplicities.is_empty() {
        let ks: PyResult<Vec<_>> = s
            .multiplicities
            .iter()
            .map(|k| {
                let e = PyDict::new(py);
                e.set_item("Eg", k.eg)?;
                e.set_item("ES", k.es)?;
                e.set_item("LP", k.lp)?;
                e.set_item("LF", k.lf)?;
                e.set_item("y", tab1_class(&k.y))?;
                Ok(e)
            })
            .collect();
        d.set_item("multiplicities", ks?)?;
    }
    if let Some(lg) = s.lg {
        d.set_item("LG", lg)?;
        d.set_item("ES_NS", s.es_ns)?;
        d.set_item("LP", s.lp)?;
        d.set_item("NT", s.nt)?;
        let ts: PyResult<Vec<_>> = s
            .transitions
            .iter()
            .map(|t| {
                let e = PyDict::new(py);
                e.set_item("ES", t.es)?;
                e.set_item("TP", t.tp)?;
                if let Some(gp) = t.gp {
                    e.set_item("GP", gp)?;
                }
                Ok(e)
            })
            .collect();
        d.set_item("transitions", ts?)?;
    }
    Ok(d)
}

fn mf13_dict<'py>(py: Python<'py>, s: &endf::mf::photon::Mf13) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("NK", s.nk)?;
    if let Some(t) = &s.sigma_total {
        d.set_item("sigma_total", tab1_class(t))?;
    }
    let photons: PyResult<Vec<_>> = s
        .photons
        .iter()
        .map(|p| {
            let e = PyDict::new(py);
            e.set_item("EG", p.eg)?;
            e.set_item("ES", p.es)?;
            e.set_item("LP", p.lp)?;
            e.set_item("LF", p.lf)?;
            e.set_item("sigma", tab1_class(&p.sigma))?;
            Ok(e)
        })
        .collect();
    d.set_item("photons", photons?)?;
    Ok(d)
}

fn mf14_dict<'py>(py: Python<'py>, s: &endf::mf::photon::Mf14) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LI", s.li)?;
    d.set_item("NK", s.nk)?;
    if let (Some(ltt), Some(ni)) = (s.ltt, s.ni) {
        d.set_item("LTT", ltt)?;
        d.set_item("NI", ni)?;
    }
    if !s.subsections.is_empty() {
        let subs: PyResult<Vec<_>> = s
            .subsections
            .iter()
            .map(|sub| {
                let e = PyDict::new(py);
                e.set_item("EG", sub.eg)?;
                e.set_item("ES", sub.es)?;
                if let Some(e_int) = &sub.e_int {
                    e.set_item("E_int", tab2_class(e_int))?;
                    e.set_item("NE", sub.ne)?;
                    e.set_item("E", sub.energy.clone())?;
                }
                if !sub.nl.is_empty() {
                    e.set_item("NL", sub.nl.clone())?;
                }
                if !sub.a_lk.is_empty() {
                    e.set_item("a_lk", sub.a_lk.clone())?;
                }
                if !sub.p_k.is_empty() {
                    e.set_item("p_k", sub.p_k.iter().map(tab1_class).collect::<Vec<_>>())?;
                }
                Ok(e)
            })
            .collect();
        d.set_item("subsections", subs?)?;
    }
    Ok(d)
}

fn mf15_dict<'py>(py: Python<'py>, s: &endf::mf::photon::Mf15) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("NC", s.nc)?;
    let subs: PyResult<Vec<_>> = s
        .subsections
        .iter()
        .map(|sub| {
            let e = PyDict::new(py);
            e.set_item("LF", sub.lf)?;
            e.set_item("p", tab1_class(&sub.p))?;
            e.set_item("E_int", tab2_class(&sub.e_int))?;
            e.set_item("NE", sub.ne)?;
            e.set_item("E", sub.energy.clone())?;
            e.set_item("g", sub.g.iter().map(tab1_class).collect::<Vec<_>>())?;
            Ok(e)
        })
        .collect();
    d.set_item("subsections", subs?)?;
    Ok(d)
}

fn mf23_dict<'py>(py: Python<'py>, s: &endf::mf::atomic::Mf23) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("EPE", s.epe)?;
    d.set_item("EFL", s.efl)?;
    d.set_item("sigma", tab1_class(&s.sigma))?;
    Ok(d)
}

fn mf27_dict<'py>(py: Python<'py>, s: &endf::mf::atomic::Mf27) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("Z", s.z)?;
    d.set_item("H", tab1_class(&s.h))?;
    Ok(d)
}

fn mf28_dict<'py>(py: Python<'py>, s: &endf::mf::atomic::Mf28) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("NSS", s.nss)?;
    let shells: PyResult<Vec<_>> = s
        .shells
        .iter()
        .map(|shell| {
            let e = PyDict::new(py);
            e.set_item("SUBI", shell.subi)?;
            e.set_item("NTR", shell.ntr)?;
            e.set_item("EBI", shell.ebi)?;
            e.set_item("ELN", shell.eln)?;
            e.set_item("SUBJ", shell.subj.clone())?;
            e.set_item("SUBK", shell.subk.clone())?;
            e.set_item("ETR", shell.etr.clone())?;
            e.set_item("FTR", shell.ftr.clone())?;
            Ok(e)
        })
        .collect();
    d.set_item("shells", shells?)?;
    Ok(d)
}

fn mf8_dict<'py>(py: Python<'py>, s: &endf::mf::mf8::Mf8) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LIS", s.lis)?;
    d.set_item("LISO", s.liso)?;
    d.set_item("NS", s.ns)?;
    d.set_item("NO", s.no)?;
    let subs: PyResult<Vec<_>> = s
        .subsections
        .iter()
        .map(|sub| {
            let e = PyDict::new(py);
            e.set_item("ZAP", sub.zap)?;
            e.set_item("ELFS", sub.elfs)?;
            e.set_item("LMF", sub.lmf)?;
            e.set_item("LFS", sub.lfs)?;
            // The decay chain of the product is present only when the
            // evaluation wrote one.
            if let Some(nd) = sub.nd {
                e.set_item("ND", nd)?;
                e.set_item("HL", sub.hl.clone())?;
                e.set_item("RTYP", sub.rtyp.clone())?;
                e.set_item("ZAN", sub.zan.clone())?;
                e.set_item("BR", sub.br.clone())?;
                e.set_item("END", sub.end.clone())?;
                e.set_item("CT", sub.ct.clone())?;
            }
            Ok(e)
        })
        .collect();
    d.set_item("subsections", subs?)?;
    Ok(d)
}

fn mf1_mt460_dict<'py>(
    py: Python<'py>,
    s: &endf::mf::mf1::Mf1Mt460,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LO", s.lo)?;
    if s.lo == 1 {
        d.set_item("NG", s.ng)?;
        d.set_item("E", s.energy.clone())?;
        d.set_item("T", s.time.iter().map(tab1_class).collect::<Vec<_>>())?;
    }
    if s.lo == 2 {
        d.set_item("lambda", s.lambda.clone())?;
    }
    Ok(d)
}

fn mf33_subsection_dict<'py>(
    py: Python<'py>,
    sub: &endf::mf::covariance::Mf33Subsection,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("XMF1", sub.xmf1)?;
    d.set_item("XLFS1", sub.xlfs1)?;
    d.set_item("MAT1", sub.mat1)?;
    d.set_item("MT1", sub.mt1)?;
    d.set_item("NC", sub.nc)?;
    d.set_item("NI", sub.ni)?;

    let nc: PyResult<Vec<_>> = sub
        .nc_subsections
        .iter()
        .map(|nc| {
            let e = PyDict::new(py);
            e.set_item("LTY", nc.lty)?;
            e.set_item("E1", nc.e1)?;
            e.set_item("E2", nc.e2)?;
            // LTY says which half of the record was written.
            if nc.lty == 0 {
                e.set_item("NCI", nc.nci)?;
                e.set_item("CI", nc.ci.clone())?;
                e.set_item("XMTI", nc.xmti.clone())?;
            } else {
                e.set_item("MATS", nc.mats)?;
                e.set_item("MTS", nc.mts)?;
                e.set_item("NEI", nc.nei)?;
                e.set_item("XMFS", nc.xmfs)?;
                e.set_item("XLFSS", nc.xlfss)?;
                e.set_item("EI", nc.ei.clone())?;
                e.set_item("WEI", nc.wei.clone())?;
            }
            Ok(e)
        })
        .collect();
    d.set_item("nc_subsections", nc?)?;

    let ni: PyResult<Vec<_>> = sub
        .ni_subsections
        .iter()
        .map(|ni| {
            let e = PyDict::new(py);
            e.set_item("LB", ni.lb)?;
            e.set_item("NT", ni.nt)?;
            // Each LB is its own record layout.
            match ni.lb {
                0..=4 => {
                    e.set_item("LT", ni.lt)?;
                    e.set_item("NP", ni.np)?;
                    e.set_item("Ek", ni.ek.clone())?;
                    e.set_item("Fk", ni.fk.clone())?;
                    e.set_item("El", ni.el.clone())?;
                    e.set_item("Fl", ni.fl.clone())?;
                }
                5 => {
                    e.set_item("LS", ni.ls)?;
                    e.set_item("NE", ni.ne)?;
                    e.set_item("Ek", ni.ek.clone())?;
                    e.set_item("Fkk", ni.fkk.clone())?;
                }
                6 => {
                    e.set_item("NER", ni.ner)?;
                    e.set_item("NEC", ni.nec)?;
                    e.set_item("ER", ni.er.clone())?;
                    e.set_item("EC", ni.ec.clone())?;
                    e.set_item("Fkl", ni.fkl.clone())?;
                }
                _ => {
                    e.set_item("LT", ni.lt)?;
                    e.set_item("NP", ni.np)?;
                    e.set_item("Ek", ni.ek.clone())?;
                    e.set_item("Fk", ni.fk.clone())?;
                }
            }
            Ok(e)
        })
        .collect();
    d.set_item("ni_subsections", ni?)?;
    Ok(d)
}

fn mf33_dict<'py>(py: Python<'py>, s: &endf::mf::covariance::Mf33) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("MTL", s.mtl)?;
    d.set_item("NL", s.nl)?;
    let subs: PyResult<Vec<_>> = s
        .subsections
        .iter()
        .map(|sub| mf33_subsection_dict(py, sub))
        .collect();
    d.set_item("subsections", subs?)?;
    Ok(d)
}

fn mf34_dict<'py>(py: Python<'py>, s: &endf::mf::covariance::Mf34) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LTT", s.ltt)?;
    d.set_item("NMT1", s.nmt1)?;
    // Always empty, matching upstream; see issue #18.
    let subs: PyResult<Vec<_>> = s
        .subsections
        .iter()
        .map(|sub| {
            let e = PyDict::new(py);
            e.set_item("MAT1", sub.mat1)?;
            e.set_item("MT1", sub.mt1)?;
            e.set_item("NL", sub.nl)?;
            e.set_item("NSS", sub.nss)?;
            e.set_item("LCT", sub.lct)?;
            e.set_item("L", sub.l.clone())?;
            e.set_item("L1", sub.l1.clone())?;
            e.set_item("NI", sub.ni.clone())?;
            let sss: PyResult<Vec<_>> = sub
                .subsubsections
                .iter()
                .map(|ss| {
                    let f = PyDict::new(py);
                    f.set_item("LS", ss.ls.clone())?;
                    f.set_item("LB", ss.lb.clone())?;
                    f.set_item("NT", ss.nt.clone())?;
                    f.set_item("NE", ss.ne.clone())?;
                    f.set_item("Data", ss.data.clone())?;
                    Ok(f)
                })
                .collect();
            e.set_item("subsubsections", sss?)?;
            Ok(e)
        })
        .collect();
    d.set_item("subsections", subs?)?;
    Ok(d)
}

fn mf40_dict<'py>(py: Python<'py>, s: &endf::mf::covariance::Mf40) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ZA", s.za)?;
    d.set_item("AWR", s.awr)?;
    d.set_item("LIS", s.lis)?;
    d.set_item("NS", s.ns)?;
    let subs: PyResult<Vec<_>> = s
        .subsections
        .iter()
        .map(|sub| {
            let e = PyDict::new(py);
            e.set_item("QM", sub.qm)?;
            e.set_item("QI", sub.qi)?;
            e.set_item("IZAP", sub.izap)?;
            e.set_item("LFS", sub.lfs)?;
            e.set_item("NL", sub.nl)?;
            let sss: PyResult<Vec<_>> = sub
                .subsubsections
                .iter()
                .map(|ss| mf33_subsection_dict(py, ss))
                .collect();
            e.set_item("subsubsections", sss?)?;
            Ok(e)
        })
        .collect();
    d.set_item("subsections", subs?)?;
    Ok(d)
}

/// One section as the Python reader's dictionary, where there is one.
///
/// `None` for a section whose dictionary shape has not been written. Those
/// are reachable through the typed accessors and the object layer; what is
/// missing is only the dictionary form the Python reader happens to use.
fn section_dict<'py>(py: Python<'py>, section: &Section) -> PyResult<Option<Bound<'py, PyDict>>> {
    Ok(Some(match section {
        Section::Mf1Mt451(s) => mf1_mt451_dict(py, s)?,
        Section::Mf1Mt452(s) => mf1_mt452_dict(py, s)?,
        Section::Mf1Mt455(s) => mf1_mt455_dict(py, s)?,
        Section::Mf1Mt460(s) => mf1_mt460_dict(py, s)?,
        Section::Mf8(s) => mf8_dict(py, s)?,
        Section::Mf3(s) => mf3_dict(py, s)?,
        Section::Mf4(s) => mf4_dict(py, s)?,
        Section::Mf5(s) => mf5_dict(py, s)?,
        Section::Mf9Mf10(s) => mf9_mf10_dict(py, s)?,
        Section::Mf12(s) => mf12_dict(py, s)?,
        Section::Mf13(s) => mf13_dict(py, s)?,
        Section::Mf14(s) => mf14_dict(py, s)?,
        Section::Mf15(s) => mf15_dict(py, s)?,
        Section::Mf23(s) => mf23_dict(py, s)?,
        Section::Mf27(s) => mf27_dict(py, s)?,
        Section::Mf28(s) => mf28_dict(py, s)?,
        Section::Mf33(s) => mf33_dict(py, s)?,
        Section::Mf34(s) => mf34_dict(py, s)?,
        Section::Mf40(s) => mf40_dict(py, s)?,
        _ => return Ok(None),
    }))
}

/// Read every material in an ENDF-6 file.
#[pyfunction]
fn get_materials(filename: &str) -> PyResult<Vec<PyMaterial>> {
    let materials = endf::materials_from_str(&read_text(filename)?).map_err(to_py_err)?;
    Ok(materials
        .into_iter()
        .map(|inner| PyMaterial { inner })
        .collect())
}

// ---------------------------------------------------------------------------
// Distributions
//
// The sum types — an angle-energy distribution, an outgoing energy law, a
// univariate density — come across as dicts tagged with a `kind` key rather
// than as a hierarchy of wrapper classes. That is the shape a consumer wants
// anyway: `kind` is exactly the discriminant column an Arrow union needs, and
// it keeps this file from growing a class per variant.
// ---------------------------------------------------------------------------

fn tab1_dict<'py>(py: Python<'py>, t: &endf::Tabulated1D) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("x", t.x.clone())?;
    d.set_item("y", t.y.clone())?;
    d.set_item("breakpoints", t.breakpoints.clone())?;
    d.set_item("interpolation", t.interpolation.clone())?;
    Ok(d)
}

fn univariate_dict<'py>(py: Python<'py>, u: &Univariate) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match u {
        Univariate::Discrete(t) => {
            d.set_item("kind", "discrete")?;
            d.set_item("x", t.x.clone())?;
            d.set_item("p", t.p.clone())?;
            d.set_item("cdf", t.cdf())?;
            if let Some(c) = &t.c {
                d.set_item("c", c.clone())?;
            }
        }
        Univariate::Tabular(t) => {
            d.set_item("kind", "tabular")?;
            d.set_item("interpolation", t.interpolation.name())?;
            d.set_item("x", t.x.clone())?;
            d.set_item("p", t.p.clone())?;
            d.set_item("cdf", t.cdf())?;
            if let Some(c) = &t.c {
                d.set_item("c", c.clone())?;
            }
        }
        Univariate::Uniform(t) => {
            d.set_item("kind", "uniform")?;
            d.set_item("a", t.a)?;
            d.set_item("b", t.b)?;
        }
        Univariate::Mixture(m) => {
            d.set_item("kind", "mixture")?;
            d.set_item("probability", m.probability.clone())?;
            let parts: PyResult<Vec<_>> = m
                .distribution
                .iter()
                .map(|sub| univariate_dict(py, sub))
                .collect();
            d.set_item("distribution", parts?)?;
        }
    }
    Ok(d)
}

fn angle_dict<'py>(py: Python<'py>, a: &AngleDistribution) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("energy", a.energy.clone())?;
    let mut mu = Vec::with_capacity(a.mu.len());
    for entry in &a.mu {
        let m = PyDict::new(py);
        match entry {
            AngleAtEnergy::Legendre(l) => {
                m.set_item("kind", "legendre")?;
                m.set_item("coefficients", l.coefficients.clone())?;
            }
            AngleAtEnergy::Tabulated(t) => {
                m.set_item("kind", "tabulated")?;
                m.set_item("f", tab1_dict(py, t)?)?;
            }
            AngleAtEnergy::Tabular(t) => {
                let inner = univariate_dict(py, &Univariate::Tabular(t.clone()))?;
                m.update(inner.as_mapping())?;
            }
            AngleAtEnergy::Isotropic(u) => {
                let inner = univariate_dict(py, &Univariate::Uniform(u.clone()))?;
                m.update(inner.as_mapping())?;
            }
        }
        mu.push(m);
    }
    d.set_item("mu", mu)?;
    Ok(d)
}

fn energy_dict<'py>(py: Python<'py>, e: &EnergyDistribution) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match e {
        EnergyDistribution::ArbitraryTabulated { energy, g, .. } => {
            d.set_item("kind", "arbitrary-tabulated")?;
            d.set_item("energy", energy.clone())?;
            let g: PyResult<Vec<_>> = g.iter().map(|t| tab1_dict(py, t)).collect();
            d.set_item("g", g?)?;
        }
        EnergyDistribution::GeneralEvaporation { u, theta, g } => {
            d.set_item("kind", "general-evaporation")?;
            d.set_item("U", *u)?;
            d.set_item("theta", tab1_dict(py, theta)?)?;
            d.set_item("g", tab1_dict(py, g)?)?;
        }
        EnergyDistribution::MaxwellEnergy { u, theta } => {
            d.set_item("kind", "maxwell")?;
            d.set_item("U", *u)?;
            d.set_item("theta", tab1_dict(py, theta)?)?;
        }
        EnergyDistribution::Evaporation { u, theta } => {
            d.set_item("kind", "evaporation")?;
            d.set_item("U", *u)?;
            d.set_item("theta", tab1_dict(py, theta)?)?;
        }
        EnergyDistribution::WattEnergy { u, a, b } => {
            d.set_item("kind", "watt")?;
            d.set_item("U", *u)?;
            d.set_item("a", tab1_dict(py, a)?)?;
            d.set_item("b", tab1_dict(py, b)?)?;
        }
        EnergyDistribution::MadlandNix { efl, efh, t_m } => {
            d.set_item("kind", "madland-nix")?;
            d.set_item("EFL", *efl)?;
            d.set_item("EFH", *efh)?;
            d.set_item("T_M", tab1_dict(py, t_m)?)?;
        }
        EnergyDistribution::LevelInelastic {
            threshold,
            mass_ratio,
        } => {
            d.set_item("kind", "level-inelastic")?;
            d.set_item("threshold", *threshold)?;
            d.set_item("mass_ratio", *mass_ratio)?;
        }
        EnergyDistribution::DiscretePhoton {
            primary_flag,
            energy,
            atomic_weight_ratio,
        } => {
            d.set_item("kind", "discrete-photon")?;
            d.set_item("primary_flag", *primary_flag)?;
            d.set_item("energy", *energy)?;
            d.set_item("atomic_weight_ratio", *atomic_weight_ratio)?;
        }
        EnergyDistribution::ContinuousTabular {
            breakpoints,
            interpolation,
            energy,
            energy_out,
        } => {
            d.set_item("kind", "continuous-tabular")?;
            d.set_item("breakpoints", breakpoints.clone())?;
            d.set_item("interpolation", interpolation.clone())?;
            d.set_item("energy", energy.clone())?;
            let out: PyResult<Vec<_>> = energy_out.iter().map(|u| univariate_dict(py, u)).collect();
            d.set_item("energy_out", out?)?;
        }
    }
    Ok(d)
}

fn angle_energy_dict<'py>(py: Python<'py>, ae: &AngleEnergy) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match ae {
        AngleEnergy::Uncorrelated(u) => {
            d.set_item("kind", "uncorrelated")?;
            if let Some(angle) = &u.angle {
                d.set_item("angle", angle_dict(py, angle)?)?;
            }
            if let Some(energy) = &u.energy {
                d.set_item("energy", energy_dict(py, energy)?)?;
            }
        }
        AngleEnergy::KalbachMann(k) => {
            d.set_item("kind", "kalbach-mann")?;
            d.set_item("breakpoints", k.breakpoints.clone())?;
            d.set_item("interpolation", k.interpolation.clone())?;
            d.set_item("energy", k.energy.clone())?;
            let out: PyResult<Vec<_>> = k
                .energy_out
                .iter()
                .map(|u| univariate_dict(py, u))
                .collect();
            d.set_item("energy_out", out?)?;
            let r: PyResult<Vec<_>> = k.precompound.iter().map(|t| tab1_dict(py, t)).collect();
            d.set_item("precompound", r?)?;
            let a: PyResult<Vec<_>> = k.slope.iter().map(|t| tab1_dict(py, t)).collect();
            d.set_item("slope", a?)?;
        }
        AngleEnergy::Correlated(c) => {
            d.set_item("kind", "correlated")?;
            d.set_item("breakpoints", c.breakpoints.clone())?;
            d.set_item("interpolation", c.interpolation.clone())?;
            d.set_item("energy", c.energy.clone())?;
            let out: PyResult<Vec<_>> = c
                .energy_out
                .iter()
                .map(|u| univariate_dict(py, u))
                .collect();
            d.set_item("energy_out", out?)?;
            let mut mu = Vec::with_capacity(c.mu.len());
            for row in &c.mu {
                let row: PyResult<Vec<_>> = row.iter().map(|u| univariate_dict(py, u)).collect();
                mu.push(row?);
            }
            d.set_item("mu", mu)?;
        }
        AngleEnergy::NBodyPhaseSpace(n) => {
            d.set_item("kind", "nbody")?;
            d.set_item("total_mass", n.total_mass)?;
            d.set_item("n_particles", n.n_particles)?;
            d.set_item("atomic_weight_ratio", n.atomic_weight_ratio)?;
            d.set_item("q_value", n.q_value)?;
        }
    }
    Ok(d)
}

// ---------------------------------------------------------------------------
// Reactions and nuclides
// ---------------------------------------------------------------------------

/// A secondary particle a reaction emits.
#[pyclass(name = "Product", module = "endf._endf")]
#[derive(Clone)]
struct PyProduct {
    inner: endf::Product,
}

#[pymethods]
impl PyProduct {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn emission_mode(&self) -> &'static str {
        self.inner.emission_mode.name()
    }

    #[getter]
    fn decay_rate(&self) -> f64 {
        self.inner.decay_rate
    }

    /// The yield, as `{"kind": "polynomial"|"tabulated", ...}`.
    #[getter]
    fn yield_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        match &self.inner.yield_ {
            endf::Yield::Polynomial(p) => {
                d.set_item("kind", "polynomial")?;
                d.set_item("coefficients", p.coefficients.clone())?;
            }
            endf::Yield::Tabulated(t) => {
                d.set_item("kind", "tabulated")?;
                d.set_item("f", tab1_dict(py, t)?)?;
            }
        }
        Ok(d)
    }

    /// The yield at an incident energy in eV.
    fn yield_at(&self, energy: f64) -> f64 {
        self.inner.yield_.eval(energy)
    }

    #[getter]
    fn applicability(&self) -> Vec<PyTabulated1D> {
        self.inner
            .applicability
            .iter()
            .map(|t| PyTabulated1D { inner: t.clone() })
            .collect()
    }

    #[getter]
    fn distribution<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .distribution
            .iter()
            .map(|d| angle_energy_dict(py, d))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Product: {}, emission={}>",
            self.inner.name,
            self.inner.emission_mode.name()
        )
    }
}

/// One reaction channel.
#[pyclass(name = "Reaction", module = "endf._endf")]
#[derive(Clone)]
struct PyReaction {
    inner: endf::Reaction,
}

#[pymethods]
impl PyReaction {
    #[getter]
    #[allow(non_snake_case)]
    fn MT(&self) -> i32 {
        self.inner.mt
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name()
    }

    #[getter]
    fn q_reaction(&self) -> f64 {
        self.inner.q_reaction
    }

    #[getter]
    fn q_massdiff(&self) -> f64 {
        self.inner.q_massdiff
    }

    #[getter]
    fn redundant(&self) -> bool {
        self.inner.redundant
    }

    #[getter]
    fn center_of_mass(&self) -> bool {
        self.inner.center_of_mass
    }

    /// Cross sections by temperature, e.g. `rx.xs["294K"]`.
    #[getter]
    fn xs(&self) -> BTreeMap<String, PyTabulated1D> {
        self.inner
            .xs
            .iter()
            .map(|(t, xs)| (t.clone(), PyTabulated1D { inner: xs.clone() }))
            .collect()
    }

    #[getter]
    fn products(&self) -> Vec<PyProduct> {
        self.inner
            .products
            .iter()
            .map(|p| PyProduct { inner: p.clone() })
            .collect()
    }

    #[getter]
    fn derived_products(&self) -> Vec<PyProduct> {
        self.inner
            .derived_products
            .iter()
            .map(|p| PyProduct { inner: p.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        match self.inner.name() {
            Some(name) => format!("<Reaction: MT={} {name}>", self.inner.mt),
            None => format!("<Reaction: MT={}>", self.inner.mt),
        }
    }
}

/// Continuous-energy neutron interaction data for one nuclide.
#[pyclass(name = "IncidentNeutron", module = "endf._endf")]
struct PyIncidentNeutron {
    inner: endf::IncidentNeutron,
}

#[pymethods]
impl PyIncidentNeutron {
    /// Read a nuclide from an ENDF evaluation.
    #[staticmethod]
    fn from_endf(material: &PyMaterial) -> PyResult<Self> {
        let inner = endf::IncidentNeutron::from_endf(&material.inner).map_err(to_py_err)?;
        Ok(PyIncidentNeutron { inner })
    }

    /// Read a nuclide from an ACE table.
    #[staticmethod]
    #[pyo3(signature = (table, metastable_scheme="mcnp"))]
    fn from_ace(table: &PyAceTable, metastable_scheme: &str) -> PyResult<Self> {
        let scheme = parse_scheme(metastable_scheme)?;
        let inner = endf::IncidentNeutron::from_ace(&table.inner, scheme).map_err(to_py_err)?;
        Ok(PyIncidentNeutron { inner })
    }

    /// Add the same nuclide at another temperature.
    #[pyo3(signature = (table, metastable_scheme="mcnp"))]
    fn add_temperature_from_ace(
        &mut self,
        table: &PyAceTable,
        metastable_scheme: &str,
    ) -> PyResult<()> {
        let scheme = parse_scheme(metastable_scheme)?;
        self.inner
            .add_temperature_from_ace(&table.inner, scheme)
            .map_err(to_py_err)
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    #[getter]
    fn atomic_number(&self) -> u32 {
        self.inner.atomic_number
    }

    #[getter]
    fn mass_number(&self) -> u32 {
        self.inner.mass_number
    }

    #[getter]
    fn metastable(&self) -> u32 {
        self.inner.metastable
    }

    #[getter]
    fn atomic_symbol(&self) -> &'static str {
        self.inner.atomic_symbol()
    }

    #[getter]
    fn atomic_weight_ratio(&self) -> Option<f64> {
        self.inner.atomic_weight_ratio
    }

    #[getter]
    #[allow(non_snake_case)]
    fn kTs(&self) -> Vec<f64> {
        self.inner.k_ts.clone()
    }

    #[getter]
    fn temperatures(&self) -> Vec<String> {
        self.inner.temperatures()
    }

    #[getter]
    fn energy(&self) -> BTreeMap<String, Vec<f64>> {
        self.inner.energy.clone()
    }

    #[getter]
    fn reactions(&self) -> BTreeMap<i32, PyReaction> {
        self.inner
            .reactions
            .iter()
            .map(|(mt, rx)| (*mt, PyReaction { inner: rx.clone() }))
            .collect()
    }

    /// What a redundant reaction is the sum of.
    fn reaction_components(&self, mt: i32) -> Vec<i32> {
        self.inner.reaction_components(mt)
    }

    /// The removal cross section at a temperature.
    #[pyo3(signature = (temperature="0K", mu_cutoff=0.0))]
    fn removal_xs(&self, temperature: &str, mu_cutoff: f64) -> PyResult<PyTabulated1D> {
        let inner = self
            .inner
            .removal_xs(temperature, mu_cutoff)
            .map_err(to_py_err)?;
        Ok(PyTabulated1D { inner })
    }

    /// The unresolved resonance probability tables, by temperature.
    #[getter]
    fn urr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (t, urr) in &self.inner.urr {
            let d = PyDict::new(py);
            d.set_item("energy", urr.energy.clone())?;
            d.set_item("table", urr.table.clone())?;
            d.set_item("shape", urr.shape.to_vec())?;
            d.set_item("interpolation", urr.interpolation)?;
            d.set_item("inelastic_flag", urr.inelastic_flag)?;
            d.set_item("absorption_flag", urr.absorption_flag)?;
            d.set_item("multiply_smooth", urr.multiply_smooth)?;
            out.set_item(t, d)?;
        }
        Ok(out)
    }

    fn __contains__(&self, mt: i32) -> bool {
        self.inner.contains(mt)
    }

    /// A reaction by MT, or by name — `n[2]`, `n["elastic"]`.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyReaction> {
        let found = if let Ok(mt) = key.extract::<i32>() {
            self.inner.get(mt)
        } else {
            let name: String = key.extract()?;
            self.inner.get_by_name(&name)
        };
        found
            .map(|rx| PyReaction { inner: rx.clone() })
            .ok_or_else(|| PyValueError::new_err(format!("no reaction {key}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "<IncidentNeutron: {}, {} reactions>",
            self.inner.name(),
            self.inner.reactions.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Photons, decay and chains
// ---------------------------------------------------------------------------

/// Photon interaction data for one element.
#[pyclass(name = "IncidentPhoton", module = "endf._endf")]
struct PyIncidentPhoton {
    inner: endf::IncidentPhoton,
}

#[pymethods]
impl PyIncidentPhoton {
    #[staticmethod]
    #[pyo3(signature = (photoatomic, relaxation=None))]
    fn from_endf(photoatomic: &PyMaterial, relaxation: Option<&PyMaterial>) -> PyResult<Self> {
        let inner =
            endf::IncidentPhoton::from_endf(&photoatomic.inner, relaxation.map(|m| &m.inner))
                .map_err(to_py_err)?;
        Ok(PyIncidentPhoton { inner })
    }

    #[staticmethod]
    fn from_ace(table: &PyAceTable) -> PyResult<Self> {
        let inner = endf::IncidentPhoton::from_ace(&table.inner).map_err(to_py_err)?;
        Ok(PyIncidentPhoton { inner })
    }

    #[getter]
    fn atomic_number(&self) -> i64 {
        self.inner.atomic_number
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Every reaction, as `{MT: {...}}`.
    #[getter]
    fn reactions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (mt, rx) in &self.inner.reactions {
            let d = PyDict::new(py);
            d.set_item("MT", rx.mt)?;
            if let Some(name) = rx.name() {
                d.set_item("name", name)?;
            }
            for (key, value) in [
                ("xs", &rx.xs),
                ("scattering_factor", &rx.scattering_factor),
                ("anomalous_real", &rx.anomalous_real),
                ("anomalous_imag", &rx.anomalous_imag),
            ] {
                if let Some(value) = value {
                    d.set_item(key, tab1_dict(py, value)?)?;
                }
            }
            if let Some(value) = rx.subshell_binding_energy {
                d.set_item("subshell_binding_energy", value)?;
            }
            if let Some(value) = rx.fluorescence_yield {
                d.set_item("fluorescence_yield", value)?;
            }
            out.set_item(mt, d)?;
        }
        Ok(out)
    }

    /// Atomic relaxation data, or None.
    #[getter]
    fn atomic_relaxation<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(r) = &self.inner.atomic_relaxation else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("subshells", r.subshells())?;
        d.set_item("binding_energy", r.binding_energy.clone())?;
        d.set_item("num_electrons", r.num_electrons.clone())?;
        let transitions = PyDict::new(py);
        for (shell, t) in &r.transitions {
            let e = PyDict::new(py);
            e.set_item("secondary_subshell", t.secondary_subshell.clone())?;
            e.set_item("tertiary_subshell", t.tertiary_subshell.clone())?;
            e.set_item("energy", t.energy.clone())?;
            e.set_item("probability", t.probability.clone())?;
            transitions.set_item(shell, e)?;
        }
        d.set_item("transitions", transitions)?;
        Ok(Some(d))
    }

    fn reaction_components(&self, mt: i32) -> Vec<i32> {
        self.inner.reaction_components(mt)
    }

    fn __contains__(&self, mt: i32) -> bool {
        self.inner.contains(mt)
    }

    fn __repr__(&self) -> String {
        format!(
            "<IncidentPhoton: {}, {} reactions>",
            self.inner.name(),
            self.inner.reactions.len()
        )
    }
}

/// Radioactive decay data for one nuclide.
#[pyclass(name = "Decay", module = "endf._endf")]
struct PyDecay {
    inner: endf::Decay,
}

#[pymethods]
impl PyDecay {
    #[staticmethod]
    fn from_endf(material: &PyMaterial) -> PyResult<Self> {
        let inner = endf::Decay::from_material(&material.inner).map_err(to_py_err)?;
        Ok(PyDecay { inner })
    }

    /// The nuclide, as a dict of its identity.
    #[getter]
    fn nuclide<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let n = &self.inner.nuclide;
        let d = PyDict::new(py);
        d.set_item("name", &n.name)?;
        d.set_item("atomic_number", n.atomic_number)?;
        d.set_item("mass_number", n.mass_number)?;
        d.set_item("isomeric_state", n.isomeric_state)?;
        d.set_item("excited_state", n.excited_state)?;
        d.set_item("mass", n.mass)?;
        d.set_item("stable", n.stable)?;
        d.set_item("spin", n.spin)?;
        d.set_item("parity", n.parity)?;
        Ok(d)
    }

    /// Half-life in seconds as `(value, uncertainty)`, or None if stable.
    #[getter]
    fn half_life(&self) -> Option<(f64, f64)> {
        self.inner.half_life
    }

    /// Decay constant in inverse seconds, or None where the half-life is
    /// unevaluated.
    #[getter]
    fn decay_constant(&self) -> Option<(f64, f64)> {
        self.inner.decay_constant()
    }

    #[getter]
    fn decay_energy(&self) -> (f64, f64) {
        self.inner.decay_energy()
    }

    #[getter]
    fn average_energies(&self) -> BTreeMap<&'static str, (f64, f64)> {
        self.inner.average_energies.clone()
    }

    /// The decay modes, each with the daughter it leaves behind.
    #[getter]
    fn modes<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .modes
            .iter()
            .map(|m| {
                let d = PyDict::new(py);
                d.set_item("parent", &m.parent)?;
                d.set_item("modes", m.modes.clone())?;
                d.set_item("daughter", m.daughter())?;
                d.set_item("daughter_state", m.daughter_state)?;
                d.set_item("energy", m.energy)?;
                d.set_item("branching_ratio", m.branching_ratio)?;
                Ok(d)
            })
            .collect()
    }

    /// What the nuclide emits, in particles per second, by particle type.
    #[getter]
    fn sources<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (particle, dist) in self.inner.sources().map_err(to_py_err)? {
            out.set_item(particle, univariate_dict(py, &dist)?)?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!("<Decay: {}>", self.inner.nuclide.name)
    }
}

/// A depletion chain.
#[pyclass(name = "Chain", module = "endf._endf")]
struct PyChain {
    inner: endf::Chain,
}

#[pymethods]
impl PyChain {
    /// Build a chain from decay, fission product yield and neutron
    /// evaluations.
    #[staticmethod]
    #[pyo3(signature = (decay, fpy, neutron, reactions=None))]
    fn from_endf(
        decay: Vec<PyRef<'_, PyMaterial>>,
        fpy: Vec<PyRef<'_, PyMaterial>>,
        neutron: Vec<PyRef<'_, PyMaterial>>,
        reactions: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let unwrap = |v: &[PyRef<'_, PyMaterial>]| -> Vec<endf::Material> {
            v.iter().map(|m| m.inner.clone()).collect()
        };
        let names: Vec<String> = reactions.unwrap_or_else(|| {
            endf::chain::DEFAULT_REACTIONS
                .iter()
                .map(|s| s.to_string())
                .collect()
        });
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let inner =
            endf::Chain::from_endf(&unwrap(&decay), &unwrap(&fpy), &unwrap(&neutron), &names)
                .map_err(to_py_err)?;
        Ok(PyChain { inner })
    }

    #[getter]
    fn nuclides<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .nuclides
            .iter()
            .map(|n| nuclide_dict(py, n))
            .collect()
    }

    /// One nuclide by name.
    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let n = self
            .inner
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("no nuclide {name}")))?;
        nuclide_dict(py, n)
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.contains(name)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The chain reachable from a set of starting nuclides.
    #[pyo3(signature = (initial, level=None))]
    fn reduce(&self, initial: Vec<String>, level: Option<usize>) -> PyChain {
        let initial: Vec<&str> = initial.iter().map(String::as_str).collect();
        PyChain {
            inner: self.inner.reduce(&initial, level),
        }
    }

    /// Everything that does not add up, nuclide by nuclide.
    #[pyo3(signature = (tolerance=1e-4))]
    fn validate(&self, tolerance: f64) -> Vec<String> {
        self.inner.validate(tolerance)
    }

    fn __repr__(&self) -> String {
        format!("<Chain: {} nuclides>", self.inner.len())
    }
}

fn nuclide_dict<'py>(py: Python<'py>, n: &endf::Nuclide) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("name", &n.name)?;
    d.set_item("half_life", n.half_life)?;
    d.set_item("decay_energy", n.decay_energy)?;
    let modes: PyResult<Vec<_>> = n
        .decay_modes
        .iter()
        .map(|m| {
            let e = PyDict::new(py);
            e.set_item("type", &m.kind)?;
            e.set_item("target", m.target.clone())?;
            e.set_item("branching_ratio", m.branching_ratio)?;
            Ok(e)
        })
        .collect();
    d.set_item("decay_modes", modes?)?;
    let reactions: PyResult<Vec<_>> = n
        .reactions
        .iter()
        .map(|r| {
            let e = PyDict::new(py);
            e.set_item("type", &r.kind)?;
            e.set_item("target", r.target.clone())?;
            e.set_item("Q", r.q_value)?;
            e.set_item("branching_ratio", r.branching_ratio)?;
            Ok(e)
        })
        .collect();
    d.set_item("reactions", reactions?)?;
    d.set_item("yield_data", n.yield_data.clone())?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// ACE
// ---------------------------------------------------------------------------

fn parse_scheme(name: &str) -> PyResult<endf::ace::MetastableScheme> {
    match name {
        "mcnp" => Ok(endf::ace::MetastableScheme::Mcnp),
        "nndc" => Ok(endf::ace::MetastableScheme::Nndc),
        other => Err(PyValueError::new_err(format!(
            "metastable_scheme must be 'mcnp' or 'nndc', got {other:?}"
        ))),
    }
}

/// One ACE cross section table.
#[pyclass(name = "AceTable", module = "endf._endf")]
#[derive(Clone)]
struct PyAceTable {
    inner: endf::ace::Table,
}

#[pymethods]
impl PyAceTable {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn atomic_weight_ratio(&self) -> f64 {
        self.inner.atomic_weight_ratio
    }

    /// Temperature in MeV, as the file stores it.
    #[getter]
    #[allow(non_snake_case)]
    fn kT(&self) -> f64 {
        self.inner.kt
    }

    /// The same, in kelvin.
    #[getter]
    fn temperature(&self) -> f64 {
        self.inner.temperature()
    }

    #[getter]
    fn zaid(&self) -> PyResult<i64> {
        self.inner.zaid().map_err(to_py_err)
    }

    /// The suffix letter, e.g. `"c"` for a continuous-energy neutron table.
    #[getter]
    fn data_type(&self) -> PyResult<String> {
        Ok(self
            .inner
            .data_type()
            .map_err(to_py_err)?
            .suffix()
            .to_string())
    }

    #[getter]
    fn nxs(&self) -> Vec<i64> {
        self.inner.nxs.clone()
    }

    #[getter]
    fn jxs(&self) -> Vec<i64> {
        self.inner.jxs.clone()
    }

    #[getter]
    fn xss(&self) -> Vec<f64> {
        self.inner.xss.clone()
    }

    fn __repr__(&self) -> String {
        format!("<AceTable: {}>", self.inner.name)
    }
}

/// Read every table in an ACE file.
#[pyfunction]
fn get_tables(filename: &str) -> PyResult<Vec<PyAceTable>> {
    let tables = endf::ace::tables_from_str(&read_text(filename)?, None).map_err(to_py_err)?;
    Ok(tables
        .into_iter()
        .map(|inner| PyAceTable { inner })
        .collect())
}

/// Read every table from the text of an ACE file.
#[pyfunction]
fn ace_tables_from_string(text: &str) -> PyResult<Vec<PyAceTable>> {
    let tables = endf::ace::tables_from_str(text, None).map_err(to_py_err)?;
    Ok(tables
        .into_iter()
        .map(|inner| PyAceTable { inner })
        .collect())
}

/// The name of a reaction, e.g. `"(n,2n)"` for MT=16.
#[pyfunction]
#[pyo3(name = "reaction_name")]
fn py_reaction_name(mt: i32) -> Option<String> {
    endf::reaction_name(mt)
}

/// The MT of a named reaction, by its own name or by an alias.
#[pyfunction]
#[pyo3(name = "reaction_mt")]
fn py_reaction_mt(name: &str) -> Option<i32> {
    endf::reaction_mt(name)
}

/// A nuclide's name in GNDS convention, e.g. `gnds_name(95, 242, 1)`.
#[pyfunction]
#[pyo3(name = "gnds_name")]
#[pyo3(signature = (z, a, m=0))]
fn py_gnds_name(z: u32, a: u32, m: u32) -> String {
    endf::gnds_name(z, a, m)
}

#[pymodule]
fn _endf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_float_endf, m)?)?;
    m.add_function(wrap_pyfunction!(py_int_endf, m)?)?;
    m.add_function(wrap_pyfunction!(get_materials, m)?)?;
    m.add_class::<PyTabulated1D>()?;
    m.add_class::<PyCrossSection>()?;
    m.add_function(wrap_pyfunction!(get_tables, m)?)?;
    m.add_function(wrap_pyfunction!(ace_tables_from_string, m)?)?;
    m.add_function(wrap_pyfunction!(py_reaction_name, m)?)?;
    m.add_function(wrap_pyfunction!(py_reaction_mt, m)?)?;
    m.add_function(wrap_pyfunction!(py_gnds_name, m)?)?;
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyProduct>()?;
    m.add_class::<PyReaction>()?;
    m.add_class::<PyIncidentNeutron>()?;
    m.add_class::<PyIncidentPhoton>()?;
    m.add_class::<PyDecay>()?;
    m.add_class::<PyChain>()?;
    m.add_class::<PyTabulated2D>()?;
    m.add_class::<PyAceTable>()?;
    Ok(())
}
