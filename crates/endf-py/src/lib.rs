//! Python bindings for the `endf` crate.
//!
//! Deliberately thin: every type here forwards to the Rust one and converts at
//! the boundary. Interpretation belongs in the Rust crate so that consumers
//! which never load Python get the same behaviour.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py_err(e: endf::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
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
struct PyMaterial {
    inner: endf::Material,
}

#[pymethods]
impl PyMaterial {
    #[new]
    fn new(filename: &str) -> PyResult<Self> {
        let inner = endf::Material::from_file(filename).map_err(to_py_err)?;
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
    let materials = endf::get_materials(filename).map_err(to_py_err)?;
    Ok(materials
        .into_iter()
        .map(|inner| PyMaterial { inner })
        .collect())
}

#[pymodule]
fn _endf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_float_endf, m)?)?;
    m.add_function(wrap_pyfunction!(py_int_endf, m)?)?;
    m.add_function(wrap_pyfunction!(get_materials, m)?)?;
    m.add_class::<PyTabulated1D>()?;
    m.add_class::<PyCrossSection>()?;
    m.add_class::<PyMaterial>()?;
    Ok(())
}
