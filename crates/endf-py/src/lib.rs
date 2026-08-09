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
    m.add_class::<PyAceTable>()?;
    Ok(())
}
