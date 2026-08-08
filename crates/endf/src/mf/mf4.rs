//! MF=4, angular distributions of secondary particles.

use crate::error::Result;
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::Reader;

/// Angular distributions given as Legendre coefficients (LTT=1).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LegendreAngles {
    /// Interpolation across incident energy.
    pub e_int: Tabulated2D,
    /// Temperature. The format gives one per incident energy; like the Python
    /// reader, only the last is kept.
    pub t: f64,
    /// Test flag, likewise the last one read.
    pub lt: i64,
    /// Incident energies.
    pub energy: Vec<f64>,
    /// Legendre coefficients at each incident energy, `a_1` upward — the
    /// `a_0 = 1` term the format omits is not inserted here.
    pub a_l: Vec<Vec<f64>>,
}

/// Angular distributions given as tabulated probabilities (LTT=2).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TabulatedAngles {
    pub e_int: Tabulated2D,
    pub t: f64,
    pub lt: i64,
    pub energy: Vec<f64>,
    /// The distribution of the scattering cosine at each incident energy.
    pub mu: Vec<Tabulated1D>,
}

/// MF=4: the angular distribution of one reaction's secondary particle.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf4 {
    pub za: i64,
    pub awr: f64,
    /// 0 isotropic, 1 Legendre, 2 tabulated, 3 Legendre then tabulated.
    pub ltt: i64,
    /// 1 when the distribution is isotropic at every energy.
    pub li: i64,
    /// 1 laboratory frame, 2 centre-of-mass.
    pub lct: i64,
    /// Present for LTT=1 and LTT=3.
    pub legendre: Option<LegendreAngles>,
    /// Present for LTT=2 and LTT=3.
    pub tabulated: Option<TabulatedAngles>,
    /// True when the obsolete energy transformation matrix was present and
    /// skipped. The Python reader warns here.
    pub had_transformation_matrix: bool,
}

fn parse_legendre(reader: &mut Reader) -> Result<LegendreAngles> {
    let tab2 = reader.tab2_record()?;
    let n_energy = tab2.cont.n2.max(0);
    let mut data = LegendreAngles {
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..n_energy {
        let list = reader.list_record()?;
        // T and LT are per incident energy in the format but only one slot is
        // kept, so the last wins. Mirrors the Python reader.
        data.t = list.cont.c1;
        data.lt = list.cont.l1;
        data.energy.push(list.cont.c2);
        data.a_l.push(list.values);
    }
    Ok(data)
}

fn parse_tabulated(reader: &mut Reader) -> Result<TabulatedAngles> {
    let tab2 = reader.tab2_record()?;
    let n_energy = tab2.cont.n2.max(0);
    let mut data = TabulatedAngles {
        e_int: tab2.table,
        ..Default::default()
    };
    for _ in 0..n_energy {
        let tab = reader.tab1_record()?;
        data.t = tab.c1;
        data.lt = tab.l1;
        data.energy.push(tab.c2);
        data.mu.push(tab.table);
    }
    Ok(data)
}

/// Parse an MF=4 section.
pub fn parse_mf4(reader: &mut Reader) -> Result<Mf4> {
    let head = reader.head_record()?;
    let lvt = head.l1;
    let ltt = head.l2;
    let c = reader.cont_record()?;
    let (li, lct, nk) = (c.l1, c.l2, c.n1);

    let mut data = Mf4 {
        za: head.za,
        awr: head.awr,
        ltt,
        li,
        lct,
        ..Default::default()
    };

    // The energy transformation matrix was removed from the format long ago.
    // Evaluations that still carry one are read past.
    if lvt > 0 {
        data.had_transformation_matrix = true;
        reader.skip_lines((nk.max(0) as usize).div_ceil(6))?;
    }

    match (ltt, li) {
        // Purely isotropic: nothing follows.
        (0, 1) => {}
        (1, 0) => data.legendre = Some(parse_legendre(reader)?),
        (2, 0) => data.tabulated = Some(parse_tabulated(reader)?),
        (3, 0) => {
            data.legendre = Some(parse_legendre(reader)?);
            data.tabulated = Some(parse_tabulated(reader)?);
        }
        _ => {}
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use crate::material::Material;

    const FIXTURE: &str = include_str!("../../../../tests/n-095_Am_244.endf");

    #[test]
    fn reads_legendre_angular_distributions() {
        let m = Material::from_str(FIXTURE).unwrap();
        let d = m.mf4(2).expect("MF=4 MT=2 is present");
        assert_eq!(d.za, 95244);
        // Elastic scattering is given in the centre-of-mass frame.
        assert_eq!(d.lct, 2);

        let leg = d
            .legendre
            .as_ref()
            .expect("LTT=1 gives Legendre coefficients");
        assert_eq!(leg.energy.len(), leg.a_l.len());
        assert!(!leg.energy.is_empty());
        assert!(leg.energy.windows(2).all(|w| w[1] >= w[0]));
    }
}
