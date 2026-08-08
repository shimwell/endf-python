//! Tabulated functions: the TAB1 and TAB2 types of the ENDF-6 format.

/// A one-dimensional tabulated function, mirroring the format's TAB1 type.
///
/// The `(x, y)` pairs are interpolated according to `interpolation`, one scheme
/// per region, where `breakpoints` holds the one-based index of each region's
/// last point.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tabulated1D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// One-based index of the last point in each interpolation region.
    pub breakpoints: Vec<i32>,
    /// ENDF interpolation scheme per region: 1 histogram, 2 linear-linear,
    /// 3 linear-log, 4 log-linear, 5 log-log.
    pub interpolation: Vec<i32>,
}

impl Tabulated1D {
    /// A function with a single linear-linear region, the format's default.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len() as i32;
        Tabulated1D {
            x,
            y,
            breakpoints: vec![n],
            interpolation: vec![2],
        }
    }

    pub fn with_regions(
        x: Vec<f64>,
        y: Vec<f64>,
        breakpoints: Vec<i32>,
        interpolation: Vec<i32>,
    ) -> Self {
        if breakpoints.is_empty() || interpolation.is_empty() {
            return Tabulated1D::new(x, y);
        }
        Tabulated1D {
            x,
            y,
            breakpoints,
            interpolation,
        }
    }

    pub fn n_pairs(&self) -> usize {
        self.x.len()
    }

    pub fn n_regions(&self) -> usize {
        self.breakpoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Evaluate the function at `x`, clamping to the tabulated range.
    pub fn eval(&self, x: f64) -> f64 {
        if self.x.is_empty() {
            return 0.0;
        }
        if x <= self.x[0] {
            return self.y[0];
        }
        let last = self.x.len() - 1;
        if x >= self.x[last] {
            return self.y[last];
        }

        // Index of the bin containing x: the count of points at or below it,
        // less one.
        let idx = self.x.partition_point(|&v| v <= x) - 1;

        // The scheme for the region this bin falls in. Regions are ordered, so
        // the first breakpoint past the bin wins; if none is, the last scheme
        // applies.
        let mut scheme = *self.interpolation.last().unwrap_or(&2);
        for (b, p) in self.breakpoints.iter().zip(&self.interpolation) {
            if (idx as i64) < (*b as i64) - 1 {
                scheme = *p;
                break;
            }
        }

        let (xi, xi1) = (self.x[idx], self.x[idx + 1]);
        let (yi, yi1) = (self.y[idx], self.y[idx + 1]);

        match scheme {
            1 => yi,
            2 => yi + (x - xi) / (xi1 - xi) * (yi1 - yi),
            3 => yi + (x / xi).ln() / (xi1 / xi).ln() * (yi1 - yi),
            4 => yi * ((x - xi) / (xi1 - xi) * (yi1 / yi).ln()).exp(),
            5 => yi * ((x / xi).ln() / (xi1 / xi).ln() * (yi1 / yi).ln()).exp(),
            _ => yi + (x - xi) / (xi1 - xi) * (yi1 - yi),
        }
    }

    /// Partial integrals from the start of the range to each tabulated point.
    ///
    /// The returned vector is the same length as `x`, beginning at zero.
    pub fn integral(&self) -> Vec<f64> {
        let n = self.x.len();
        if n == 0 {
            return Vec::new();
        }
        let mut partial = vec![0.0; n.saturating_sub(1)];

        let mut i_low = 0usize;
        for (k, &b) in self.breakpoints.iter().enumerate() {
            let i_high = ((b as i64 - 1).max(0) as usize).min(n.saturating_sub(1));
            let scheme = *self.interpolation.get(k).unwrap_or(&2);
            // The bin index reads x and y at both i and i+1, so it is the index
            // itself that is wanted here, not an iterator over one of them.
            #[allow(clippy::needless_range_loop)]
            for i in i_low..i_high {
                let (x0, x1) = (self.x[i], self.x[i + 1]);
                let (y0, y1) = (self.y[i], self.y[i + 1]);
                partial[i] = match scheme {
                    1 => y0 * (x1 - x0),
                    2 => {
                        let m = (y1 - y0) / (x1 - x0);
                        (y0 - m * x0) * (x1 - x0) + m * (x1 * x1 - x0 * x0) / 2.0
                    }
                    3 => {
                        let logx = (x1 / x0).ln();
                        let m = (y1 - y0) / logx;
                        y0 + m * (x1 * (logx - 1.0) + x0)
                    }
                    4 => {
                        let m = (y1 / y0).ln() / (x1 - x0);
                        y0 / m * ((m * (x1 - x0)).exp() - 1.0)
                    }
                    5 => {
                        let m = (y1 / y0).ln() / (x1 / x0).ln();
                        y0 / ((m + 1.0) * x0.powf(m)) * (x1.powf(m + 1.0) - x0.powf(m + 1.0))
                    }
                    _ => 0.0,
                };
            }
            i_low = i_high;
        }

        let mut out = Vec::with_capacity(n);
        out.push(0.0);
        let mut running = 0.0;
        for v in partial {
            running += v;
            out.push(running);
        }
        out
    }
}

/// Interpolation metadata for a two-dimensional function, the format's TAB2.
///
/// TAB2 carries no values of its own: it describes how to interpolate between
/// the subrecords that follow it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tabulated2D {
    pub breakpoints: Vec<i32>,
    pub interpolation: Vec<i32>,
}

/// A polynomial in ascending powers, used where the format gives coefficients
/// rather than a table (nu-bar in MF=1, for instance).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polynomial {
    pub coefficients: Vec<f64>,
}

impl Polynomial {
    pub fn new(coefficients: Vec<f64>) -> Self {
        Polynomial { coefficients }
    }

    pub fn eval(&self, x: f64) -> f64 {
        // Horner, from the highest power down.
        self.coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc * x + c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_linear_matches_the_docstring_example() {
        // >>> f = Tabulated1D([0, 10], [4, 5])
        // [4.0, 4.25, 4.5, 4.75, 5.0]
        let f = Tabulated1D::new(vec![0.0, 10.0], vec![4.0, 5.0]);
        let got: Vec<f64> = [0.0, 2.5, 5.0, 7.5, 10.0]
            .iter()
            .map(|&x| f.eval(x))
            .collect();
        for (a, b) in got.iter().zip([4.0, 4.25, 4.5, 4.75, 5.0]) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn clamps_outside_the_tabulated_range() {
        let f = Tabulated1D::new(vec![1.0, 2.0], vec![10.0, 20.0]);
        assert_eq!(f.eval(-5.0), 10.0);
        assert_eq!(f.eval(99.0), 20.0);
    }

    #[test]
    fn histogram_holds_the_left_value() {
        let f =
            Tabulated1D::with_regions(vec![0.0, 1.0, 2.0], vec![3.0, 7.0, 9.0], vec![3], vec![1]);
        assert_eq!(f.eval(0.5), 3.0);
        assert_eq!(f.eval(1.5), 7.0);
    }

    #[test]
    fn integral_of_a_constant_is_the_width() {
        let f = Tabulated1D::new(vec![0.0, 1.0, 2.0], vec![5.0, 5.0, 5.0]);
        let it = f.integral();
        assert_eq!(it.len(), 3);
        assert!((it[0] - 0.0).abs() < 1e-12);
        assert!((it[1] - 5.0).abs() < 1e-12);
        assert!((it[2] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn polynomial_evaluates_in_ascending_powers() {
        // 1 + 2x + 3x^2 at x = 2 -> 17
        let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.eval(2.0), 17.0);
    }
}
