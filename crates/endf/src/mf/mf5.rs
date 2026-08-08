//! MF=5, energy distributions of secondary particles.

use crate::error::{Error, Result};
use crate::function::{Tabulated1D, Tabulated2D};
use crate::records::Reader;

/// MF=5: the energy distributions of one reaction.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Mf5 {
    pub za: i64,
    pub awr: f64,
    pub nk: i64,
    pub subsections: Vec<Subsection>,
}

/// One partial distribution, with the fraction of emission it accounts for.
#[derive(Debug, Clone, PartialEq)]
pub struct Subsection {
    /// Which law the distribution uses.
    pub lf: i64,
    /// Applicability: the fraction of the emission this subsection describes,
    /// as a function of incident energy.
    pub p: Tabulated1D,
    pub distribution: EnergyDistribution,
}

/// An energy distribution, in whichever law the evaluation uses.
#[derive(Debug, Clone, PartialEq)]
pub enum EnergyDistribution {
    /// LF=1: an arbitrary tabulated function of outgoing energy.
    ArbitraryTabulated {
        e_int: Tabulated2D,
        /// Incident energies.
        energy: Vec<f64>,
        /// The outgoing distribution at each incident energy.
        g: Vec<Tabulated1D>,
    },
    /// LF=5: general evaporation spectrum.
    GeneralEvaporation {
        u: f64,
        theta: Tabulated1D,
        g: Tabulated1D,
    },
    /// LF=7: simple Maxwellian fission spectrum.
    MaxwellEnergy { u: f64, theta: Tabulated1D },
    /// LF=9: evaporation spectrum.
    Evaporation { u: f64, theta: Tabulated1D },
    /// LF=11: energy-dependent Watt spectrum.
    WattEnergy {
        u: f64,
        a: Tabulated1D,
        b: Tabulated1D,
    },
    /// LF=12: Madland-Nix fission spectrum.
    MadlandNix {
        efl: f64,
        efh: f64,
        t_m: Tabulated1D,
    },
}

/// Parse an MF=5 section.
pub fn parse_mf5(reader: &mut Reader) -> Result<Mf5> {
    let head = reader.head_record()?;
    let mut data = Mf5 {
        za: head.za,
        awr: head.awr,
        nk: head.n1,
        subsections: Vec::new(),
    };

    for _ in 0..data.nk.max(0) {
        // The applicability record also carries the law and its parameters.
        let applicability = reader.tab1_record()?;
        let lf = applicability.l2;
        let (c1, c2) = (applicability.c1, applicability.c2);

        let distribution = match lf {
            1 => {
                let tab2 = reader.tab2_record()?;
                let n = tab2.cont.n2.max(0);
                let mut energy = Vec::with_capacity(n as usize);
                let mut g = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    let tab = reader.tab1_record()?;
                    energy.push(tab.c2);
                    g.push(tab.table);
                }
                EnergyDistribution::ArbitraryTabulated {
                    e_int: tab2.table,
                    energy,
                    g,
                }
            }
            5 => EnergyDistribution::GeneralEvaporation {
                u: c1,
                theta: reader.tab1_record()?.table,
                g: reader.tab1_record()?.table,
            },
            7 => EnergyDistribution::MaxwellEnergy {
                u: c1,
                theta: reader.tab1_record()?.table,
            },
            9 => EnergyDistribution::Evaporation {
                u: c1,
                theta: reader.tab1_record()?.table,
            },
            11 => EnergyDistribution::WattEnergy {
                u: c1,
                a: reader.tab1_record()?.table,
                b: reader.tab1_record()?.table,
            },
            12 => EnergyDistribution::MadlandNix {
                efl: c1,
                efh: c2,
                t_m: reader.tab1_record()?.table,
            },
            // The Python reader leaves `dist` unbound here and raises
            // UnboundLocalError; this reports the same failure with a usable
            // message.
            _ => {
                return Err(Error::Unsupported {
                    what: "an unrecognised MF=5 energy distribution law",
                })
            }
        };

        data.subsections.push(Subsection {
            lf,
            p: applicability.table,
            distribution,
        });
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    const FIXTURE: &str = include_str!("../../../../tests/n-095_Am_244.endf");

    #[test]
    fn reads_the_fission_neutron_spectrum() {
        let m = Material::from_str(FIXTURE).unwrap();
        let d = m.mf5(18).expect("MF=5 MT=18 is present");
        assert_eq!(d.za, 95244);
        assert_eq!(d.subsections.len(), d.nk as usize);
        assert!(!d.subsections.is_empty());

        // The applicability of a single subsection is unity throughout.
        let sub = &d.subsections[0];
        assert!(sub.p.y.iter().all(|&v| v == 1.0));

        // Am244 gives its fission spectrum as a simple Maxwellian.
        assert_eq!(sub.lf, 7);
        match &sub.distribution {
            EnergyDistribution::MaxwellEnergy { u, theta } => {
                // U bounds the outgoing energy by 0 <= E' <= E - U, so a
                // negative U widens that range rather than narrowing it. This
                // evaluation uses -20 MeV.
                assert_eq!(*u, -2.0e7);
                // Theta is the Maxwellian temperature in eV, constant here.
                assert!(!theta.x.is_empty());
                assert!(theta.y.iter().all(|&v| v > 0.0));
            }
            other => panic!("unexpected distribution {other:?}"),
        }
    }
}
