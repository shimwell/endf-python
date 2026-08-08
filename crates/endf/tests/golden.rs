//! Hold the Rust reader to what the Python reader produces.
//!
//! Every file in `tests/golden/` is a reference dump written by
//! `tools/dump_golden.py`. This test finds each one, reads the ENDF file it
//! names, builds the same `path -> value` map from its own parse, and compares
//! the two maps whole.
//!
//! Comparing maps rather than walking records is what makes this scale to the
//! whole package: a field that is renamed, dropped or added shows up as a path
//! on one side and not the other, without the test needing to know what the
//! field was for.
//!
//! Values are compared exactly. The dump records the shortest round-tripping
//! decimal and both readers parse decimals with correct rounding, so any
//! difference is a real one. The single exception is `…/evaly`, the sampled
//! interpolation, where the two languages evaluate the same expression through
//! their own `ln` and `exp`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use endf::mf::mf1::{FissionEnergyRelease, Nu, FISSION_ENERGY_COMPONENTS};
use endf::mf::mf2::{ResonanceParameters, UnresolvedParameters};
use endf::mf::mf5::EnergyDistribution;
use endf::mf::mf6::Distribution as Mf6Distribution;
use endf::{materials_from_str, Material, Section, Tabulated1D, Tabulated2D};

/// Mirrors `MAX_SAMPLES` in `tools/dump_golden.py`.
const MAX_SAMPLES: usize = 24;

/// Relative tolerance for sampled interpolation. Everything else is exact.
const EVAL_TOL: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Floats(Vec<f64>),
    Ints(Vec<i64>),
    Text(String),
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Value::Floats(_) => "F",
            Value::Ints(_) => "I",
            Value::Text(_) => "T",
        }
    }
}

#[derive(Default)]
struct Dump {
    map: BTreeMap<String, Value>,
}

impl Dump {
    fn put(&mut self, path: String, value: Value) {
        if let Some(old) = self.map.insert(path.clone(), value) {
            panic!("the Rust dump wrote {path} twice (previously {old:?})");
        }
    }

    fn floats(&mut self, path: String, values: Vec<f64>) {
        self.put(path, Value::Floats(values));
    }

    fn float(&mut self, path: String, value: f64) {
        self.put(path, Value::Floats(vec![value]));
    }

    fn ints(&mut self, path: String, values: Vec<i64>) {
        self.put(path, Value::Ints(values));
    }

    fn int(&mut self, path: String, value: i64) {
        self.put(path, Value::Ints(vec![value]));
    }

    fn text(&mut self, path: String, value: &str) {
        self.put(path, Value::Text(value.to_string()));
    }

    fn tab1(&mut self, path: &str, t: &Tabulated1D) {
        self.floats(format!("{path}/x"), t.x.clone());
        self.floats(format!("{path}/y"), t.y.clone());
        self.ints(
            format!("{path}/bp"),
            t.breakpoints.iter().map(|&v| v as i64).collect(),
        );
        self.ints(
            format!("{path}/int"),
            t.interpolation.iter().map(|&v| v as i64).collect(),
        );
        let points = sample_points(t);
        if !points.is_empty() {
            let values = points.iter().map(|&p| t.eval(p)).collect();
            self.floats(format!("{path}/evalx"), points);
            self.floats(format!("{path}/evaly"), values);
        }
    }

    fn tab2(&mut self, path: &str, t: &Tabulated2D) {
        self.ints(
            format!("{path}/bp"),
            t.breakpoints.iter().map(|&v| v as i64).collect(),
        );
        self.ints(
            format!("{path}/int"),
            t.interpolation.iter().map(|&v| v as i64).collect(),
        );
    }
}

/// Mirrors `sample_points` in `tools/dump_golden.py`, index for index. The
/// arithmetic is the same in both languages, so the abscissae come out
/// bit-identical and only the ordinates need a tolerance.
fn sample_points(t: &Tabulated1D) -> Vec<f64> {
    let x = &t.x;
    if x.len() < 2 {
        return x.clone();
    }
    let n_bins = x.len() - 1;

    let mut wanted: BTreeSet<usize> = BTreeSet::new();
    for &b in &t.breakpoints {
        for i in [b as i64 - 2, b as i64 - 1] {
            if i >= 0 && (i as usize) < n_bins {
                wanted.insert(i as usize);
            }
        }
    }
    let step = (n_bins / MAX_SAMPLES).max(1);
    wanted.extend((0..n_bins).step_by(step));

    let mut points = Vec::with_capacity(wanted.len() + 4);
    points.push(x[0] * 0.5);
    points.push(x[0]);
    points.extend(wanted.iter().map(|&i| 0.5 * (x[i] + x[i + 1])));
    points.push(x[x.len() - 1]);
    points.push(x[x.len() - 1] * 2.0);
    points
}

// --------------------------------------------------------------------------
// One dumper per ENDF file, mirroring tools/dump_golden.py.
// --------------------------------------------------------------------------

fn dump_nu(d: &mut Dump, path: &str, nu: &Nu) {
    match nu {
        Nu::Polynomial(c) => d.floats(format!("{path}/poly"), c.clone()),
        Nu::Tabulated(t) => d.tab1(&format!("{path}/tab"), t),
        Nu::Absent => {}
    }
}

fn dump_mf2_parameters(d: &mut Dump, rp: &str, p: &ResonanceParameters) {
    match p {
        ResonanceParameters::ScatteringRadius { spi, ap, nls } => {
            d.float(format!("{rp}/SPI"), *spi);
            d.float(format!("{rp}/AP"), *ap);
            d.int(format!("{rp}/NLS"), *nls);
        }
        ResonanceParameters::BreitWigner(b) => {
            if let Some(ape) = &b.ape {
                d.tab1(&format!("{rp}/APE"), ape);
            }
            d.float(format!("{rp}/SPI"), b.spi);
            d.float(format!("{rp}/AP"), b.ap);
            d.int(format!("{rp}/NLS"), b.nls);
            for (i, s) in b.sections.iter().enumerate() {
                let sp = format!("{rp}/sections/{i}");
                d.float(format!("{sp}/AWRI"), s.awri);
                d.float(format!("{sp}/QX"), s.qx);
                d.int(format!("{sp}/L"), s.l);
                d.int(format!("{sp}/LRX"), s.lrx);
                d.int(format!("{sp}/NRS"), s.nrs);
                d.floats(format!("{sp}/ER"), s.er.clone());
                d.floats(format!("{sp}/AJ"), s.aj.clone());
                d.floats(format!("{sp}/GT"), s.gt.clone());
                d.floats(format!("{sp}/GN"), s.gn.clone());
                d.floats(format!("{sp}/GG"), s.gg.clone());
                d.floats(format!("{sp}/GF"), s.gf.clone());
            }
        }
        ResonanceParameters::ReichMoore(r) => {
            if let Some(ape) = &r.ape {
                d.tab1(&format!("{rp}/APE"), ape);
            }
            d.float(format!("{rp}/SPI"), r.spi);
            d.float(format!("{rp}/AP"), r.ap);
            d.int(format!("{rp}/LAD"), r.lad);
            d.int(format!("{rp}/NLS"), r.nls);
            d.int(format!("{rp}/NLSC"), r.nlsc);
            for (i, s) in r.sections.iter().enumerate() {
                let sp = format!("{rp}/sections/{i}");
                d.float(format!("{sp}/AWRI"), s.awri);
                d.float(format!("{sp}/APL"), s.apl);
                d.int(format!("{sp}/L"), s.l);
                d.int(format!("{sp}/NRS"), s.nrs);
                d.floats(format!("{sp}/ER"), s.er.clone());
                d.floats(format!("{sp}/AJ"), s.aj.clone());
                d.floats(format!("{sp}/GN"), s.gn.clone());
                d.floats(format!("{sp}/GG"), s.gg.clone());
                d.floats(format!("{sp}/GFA"), s.gfa.clone());
                d.floats(format!("{sp}/GFB"), s.gfb.clone());
            }
        }
        ResonanceParameters::RMatrixLimited(r) => {
            d.int(format!("{rp}/IFG"), r.ifg);
            d.int(format!("{rp}/KRM"), r.krm);
            d.int(format!("{rp}/NJS"), r.njs);
            d.int(format!("{rp}/KRL"), r.krl);
            d.int(format!("{rp}/NPP"), r.npp);
            let pp = &r.particle_pairs;
            for (key, values) in [
                ("MA", &pp.ma),
                ("MB", &pp.mb),
                ("ZA", &pp.za),
                ("ZB", &pp.zb),
                ("IA", &pp.ia),
                ("IB", &pp.ib),
                ("Q", &pp.q),
                ("PNT", &pp.pnt),
                ("SHF", &pp.shf),
                ("MT", &pp.mt),
                ("PA", &pp.pa),
                ("PB", &pp.pb),
            ] {
                d.floats(format!("{rp}/particle_pairs/{key}"), values.clone());
            }
            for (i, g) in r.spin_groups.iter().enumerate() {
                let gp = format!("{rp}/spin_groups/{i}");
                d.float(format!("{gp}/AJ"), g.aj);
                d.float(format!("{gp}/PJ"), g.pj);
                d.int(format!("{gp}/KBK"), g.kbk);
                d.int(format!("{gp}/KPS"), g.kps);
                d.int(format!("{gp}/NCH"), g.nch);
                d.int(format!("{gp}/NRS"), g.nrs);
                d.int(format!("{gp}/NX"), g.nx);
                let ch = &g.channels;
                for (key, values) in [
                    ("PPI", &ch.ppi),
                    ("L", &ch.l),
                    ("SCH", &ch.sch),
                    ("BND", &ch.bnd),
                    ("APE", &ch.ape),
                    ("APT", &ch.apt),
                ] {
                    d.floats(format!("{gp}/channels/{key}"), values.clone());
                }
                d.floats(format!("{gp}/ER"), g.er.clone());
                for (c, row) in g.gam.iter().enumerate() {
                    d.floats(format!("{gp}/GAM/{c}"), row.clone());
                }
                for (key, value) in [("LCH", g.lch), ("LBK", g.lbk), ("LPS", g.lps)] {
                    if let Some(v) = value {
                        d.int(format!("{gp}/{key}"), v);
                    }
                }
                for (key, value) in [("ED", g.ed), ("EU", g.eu)] {
                    if let Some(v) = value {
                        d.float(format!("{gp}/{key}"), v);
                    }
                }
                for (key, table) in [
                    ("RBR", &g.rbr),
                    ("RBI", &g.rbi),
                    ("PSR", &g.psr),
                    ("PSI", &g.psi),
                ] {
                    if let Some(t) = table {
                        d.tab1(&format!("{gp}/{key}"), t);
                    }
                }
            }
        }
        ResonanceParameters::Unresolved(u) => {
            if let Some(ape) = &u.ape {
                d.tab1(&format!("{rp}/APE"), ape);
            }
            d.float(format!("{rp}/SPI"), u.spi);
            d.float(format!("{rp}/AP"), u.ap);
            d.int(format!("{rp}/LSSF"), u.lssf);
            d.int(format!("{rp}/NLS"), u.nls);
            if let Some(ne) = u.ne {
                d.int(format!("{rp}/NE"), ne);
                d.floats(format!("{rp}/ES"), u.es.clone());
            }
            for (i, r) in u.ranges.iter().enumerate() {
                let up = format!("{rp}/ranges/{i}");
                d.float(format!("{up}/AWRI"), r.awri);
                d.int(format!("{up}/L"), r.l);
                d.int(format!("{up}/NJS"), r.njs);
                if !r.d.is_empty() {
                    d.floats(format!("{up}/D"), r.d.clone());
                    d.floats(format!("{up}/AJ"), r.aj.clone());
                    d.floats(format!("{up}/AMUN"), r.amun.clone());
                    d.floats(format!("{up}/GNO"), r.gno.clone());
                    d.floats(format!("{up}/GG"), r.gg.clone());
                }
                for (j, p) in r.parameters.iter().enumerate() {
                    let pp = format!("{up}/parameters/{j}");
                    match p {
                        UnresolvedParameters::CaseB {
                            muf,
                            d: dd,
                            aj,
                            amun,
                            gn0,
                            gg,
                            gf,
                        } => {
                            d.int(format!("{pp}/MUF"), *muf);
                            d.float(format!("{pp}/D"), *dd);
                            d.float(format!("{pp}/AJ"), *aj);
                            d.float(format!("{pp}/AMUN"), *amun);
                            d.float(format!("{pp}/GN0"), *gn0);
                            d.float(format!("{pp}/GG"), *gg);
                            d.floats(format!("{pp}/GF"), gf.clone());
                        }
                        UnresolvedParameters::CaseC {
                            aj,
                            interpolation,
                            ne,
                            amux,
                            amun,
                            amuf,
                            e,
                            d: dd,
                            gx,
                            gn0,
                            gg,
                            gf,
                        } => {
                            d.float(format!("{pp}/AJ"), *aj);
                            d.int(format!("{pp}/INT"), *interpolation);
                            d.int(format!("{pp}/NE"), *ne);
                            d.float(format!("{pp}/AMUX"), *amux);
                            d.float(format!("{pp}/AMUN"), *amun);
                            d.float(format!("{pp}/AMUF"), *amuf);
                            d.floats(format!("{pp}/E"), e.clone());
                            d.floats(format!("{pp}/D"), dd.clone());
                            d.floats(format!("{pp}/GX"), gx.clone());
                            d.floats(format!("{pp}/GN0"), gn0.clone());
                            d.floats(format!("{pp}/GG"), gg.clone());
                            d.floats(format!("{pp}/GF"), gf.clone());
                        }
                    }
                }
            }
        }
        ResonanceParameters::Absent => {}
    }
}

fn dump_mf6_distribution(d: &mut Dump, dp: &str, dist: &Mf6Distribution) {
    match dist {
        Mf6Distribution::None => {}
        Mf6Distribution::ContinuumEnergyAngle(c) => {
            d.int(format!("{dp}/LANG"), c.lang);
            d.int(format!("{dp}/LEP"), c.lep);
            d.int(format!("{dp}/NR"), c.nr);
            d.int(format!("{dp}/NE"), c.ne);
            d.tab2(&format!("{dp}/E_int"), &c.e_int);
            d.floats(format!("{dp}/E"), c.energy.clone());
            for (j, s) in c.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.int(format!("{sp}/ND"), s.nd);
                d.int(format!("{sp}/NA"), s.na);
                d.int(format!("{sp}/NW"), s.nw);
                d.int(format!("{sp}/NEP"), s.nep);
                d.floats(format!("{sp}/Eout"), s.e_out.clone());
                for (r, row) in s.b.iter().enumerate() {
                    d.floats(format!("{sp}/b/{r}"), row.clone());
                }
            }
        }
        Mf6Distribution::DiscreteTwoBody(t) => {
            d.int(format!("{dp}/NR"), t.nr);
            d.int(format!("{dp}/NE"), t.ne);
            d.tab2(&format!("{dp}/E_int"), &t.e_int);
            d.floats(format!("{dp}/E"), t.energy.clone());
            for (j, s) in t.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.int(format!("{sp}/LANG"), s.lang);
                d.int(format!("{sp}/NW"), s.nw);
                d.int(format!("{sp}/NL"), s.nl);
                d.floats(format!("{sp}/A_l"), s.a_l.clone());
            }
        }
        Mf6Distribution::ChargedParticleElastic(c) => {
            d.float(format!("{dp}/SPI"), c.spi);
            d.int(format!("{dp}/LIDP"), c.lidp);
            d.int(format!("{dp}/NE"), c.ne);
            d.tab2(&format!("{dp}/E_int"), &c.e_int);
            for (j, s) in c.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.float(format!("{sp}/E"), s.energy);
                d.int(format!("{sp}/LTP"), s.ltp);
                d.int(format!("{sp}/NW"), s.nw);
                d.int(format!("{sp}/NL"), s.nl);
                d.floats(format!("{sp}/A"), s.a.clone());
            }
        }
        Mf6Distribution::NBodyPhaseSpace { apsx, npsx } => {
            d.float(format!("{dp}/APSX"), *apsx);
            d.int(format!("{dp}/NPSX"), *npsx);
        }
        Mf6Distribution::LaboratoryAngleEnergy(l) => {
            d.int(format!("{dp}/NE"), l.ne);
            d.tab2(&format!("{dp}/E_int"), &l.e_int);
            for (j, s) in l.distribution.iter().enumerate() {
                let sp = format!("{dp}/distribution/{j}");
                d.float(format!("{sp}/E"), s.energy);
                d.int(format!("{sp}/NRM"), s.nrm);
                d.int(format!("{sp}/NMU"), s.nmu);
                d.tab2(&format!("{sp}/mu_int"), &s.mu_int);
                for (k, entry) in s.mu.iter().enumerate() {
                    d.float(format!("{sp}/mu/{k}/mu"), entry.mu);
                    d.tab1(&format!("{sp}/mu/{k}/f"), &entry.f);
                }
            }
        }
    }
}

fn dump_section(d: &mut Dump, path: &str, section: &Section) {
    match section {
        Section::Mf1Mt451(s) => {
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
                d.int(format!("{path}/{key}"), value);
            }
            for (key, value) in [
                ("AWR", s.awr),
                ("ELIS", s.elis),
                ("STA", s.sta),
                ("AWI", s.awi),
                ("EMAX", s.emax),
                ("TEMP", s.temp),
            ] {
                d.float(format!("{path}/{key}"), value);
            }
            if let Some(zsymam) = &s.zsymam {
                d.text(format!("{path}/ZSYMAM"), zsymam);
                for (key, value) in [
                    ("ALAB", &s.alab),
                    ("EDATE", &s.edate),
                    ("AUTH", &s.auth),
                    ("REF", &s.reference),
                    ("DDATE", &s.ddate),
                    ("RDATE", &s.rdate),
                    ("ENDATE", &s.endate),
                ] {
                    d.text(format!("{path}/{key}"), value.as_deref().unwrap_or(""));
                }
                for (i, line) in s.hsub.iter().enumerate() {
                    d.text(format!("{path}/HSUB/{i}"), line);
                }
                for (i, line) in s.description.iter().enumerate() {
                    d.text(format!("{path}/description/{i}"), line);
                }
            }
            for (i, &(mf, mt, nc, md)) in s.section_list.iter().enumerate() {
                d.ints(format!("{path}/section_list/{i}"), vec![mf, mt, nc, md]);
            }
        }

        Section::Mf1Mt452(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LNU"), s.lnu);
            dump_nu(d, path, &s.nu);
        }

        Section::Mf1Mt455(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LDG"), s.ldg);
            d.int(format!("{path}/LNU"), s.lnu);
            if s.ldg == 0 {
                d.floats(format!("{path}/lambda"), s.lambda.clone());
            }
            if let Some(e_int) = &s.e_int {
                d.tab2(&format!("{path}/E_int"), e_int);
            }
            for (i, c) in s.constants.iter().enumerate() {
                d.float(format!("{path}/constants/{i}/E"), c.energy);
                d.floats(format!("{path}/constants/{i}/lambda"), c.lambda.clone());
                d.floats(format!("{path}/constants/{i}/alpha"), c.alpha.clone());
            }
            dump_nu(d, path, &s.nu);
        }

        Section::Mf1Mt458(s) => {
            d.float(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LFC"), s.lfc);
            d.int(format!("{path}/NPLY"), s.nply);
            if s.lfc == 1 {
                d.int(format!("{path}/NFC"), s.nfc);
            }
            for (name, component) in FISSION_ENERGY_COMPONENTS.iter().zip(&s.components) {
                match component {
                    FissionEnergyRelease::Polynomial(pairs) => {
                        let flat = pairs.iter().flat_map(|&(c, u)| [c, u]).collect();
                        d.floats(format!("{path}/{name}/poly"), flat);
                    }
                    FissionEnergyRelease::Tabulated { ldrv, eifc } => {
                        d.int(format!("{path}/{name}/LDRV"), *ldrv);
                        d.tab1(&format!("{path}/{name}/EIFC"), eifc);
                    }
                }
            }
        }

        Section::Mf1Mt460(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LO"), s.lo);
            if s.lo == 1 {
                d.int(format!("{path}/NG"), s.ng);
                d.floats(format!("{path}/E"), s.energy.clone());
            }
            for (i, t) in s.time.iter().enumerate() {
                d.tab1(&format!("{path}/T/{i}"), t);
            }
            if s.lo == 2 {
                d.floats(format!("{path}/lambda"), s.lambda.clone());
            }
        }

        Section::Mf2(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NIS"), s.nis);
            for (a, iso) in s.isotopes.iter().enumerate() {
                let ip = format!("{path}/isotopes/{a}");
                d.float(format!("{ip}/ZAI"), iso.zai);
                d.float(format!("{ip}/ABN"), iso.abn);
                d.int(format!("{ip}/LFW"), iso.lfw);
                d.int(format!("{ip}/NER"), iso.ner);
                for (b, r) in iso.ranges.iter().enumerate() {
                    let rp = format!("{ip}/ranges/{b}");
                    d.float(format!("{rp}/EL"), r.el);
                    d.float(format!("{rp}/EH"), r.eh);
                    d.int(format!("{rp}/LRU"), r.lru);
                    d.int(format!("{rp}/LRF"), r.lrf);
                    d.int(format!("{rp}/NRO"), r.nro);
                    d.int(format!("{rp}/NAPS"), r.naps);
                    dump_mf2_parameters(d, &rp, &r.parameters);
                }
            }
        }

        Section::Mf3(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.float(format!("{path}/QM"), s.qm);
            d.float(format!("{path}/QI"), s.qi);
            d.int(format!("{path}/LR"), s.lr);
            d.tab1(&format!("{path}/sigma"), &s.sigma);
        }

        Section::Mf4(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/LTT"), s.ltt);
            d.int(format!("{path}/LI"), s.li);
            d.int(format!("{path}/LCT"), s.lct);
            if let Some(l) = &s.legendre {
                let sp = format!("{path}/legendre");
                d.tab2(&format!("{sp}/E_int"), &l.e_int);
                d.float(format!("{sp}/T"), l.t);
                d.int(format!("{sp}/LT"), l.lt);
                d.floats(format!("{sp}/E"), l.energy.clone());
                for (i, a) in l.a_l.iter().enumerate() {
                    d.floats(format!("{sp}/a_l/{i}"), a.clone());
                }
            }
            if let Some(t) = &s.tabulated {
                let sp = format!("{path}/tabulated");
                d.tab2(&format!("{sp}/E_int"), &t.e_int);
                d.float(format!("{sp}/T"), t.t);
                d.int(format!("{sp}/LT"), t.lt);
                d.floats(format!("{sp}/E"), t.energy.clone());
                for (i, mu) in t.mu.iter().enumerate() {
                    d.tab1(&format!("{sp}/mu/{i}"), mu);
                }
            }
        }

        Section::Mf5(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/NK"), s.nk);
            for (i, sub) in s.subsections.iter().enumerate() {
                let sp = format!("{path}/subsections/{i}");
                d.int(format!("{sp}/LF"), sub.lf);
                d.tab1(&format!("{sp}/p"), &sub.p);
                let dp = format!("{sp}/distribution");
                match &sub.distribution {
                    EnergyDistribution::ArbitraryTabulated { e_int, energy, g } => {
                        d.tab2(&format!("{dp}/E_int"), e_int);
                        d.floats(format!("{dp}/E"), energy.clone());
                        for (j, t) in g.iter().enumerate() {
                            d.tab1(&format!("{dp}/g/{j}"), t);
                        }
                    }
                    EnergyDistribution::GeneralEvaporation { u, theta, g } => {
                        d.float(format!("{dp}/U"), *u);
                        d.tab1(&format!("{dp}/theta"), theta);
                        d.tab1(&format!("{dp}/g"), g);
                    }
                    EnergyDistribution::MaxwellEnergy { u, theta }
                    | EnergyDistribution::Evaporation { u, theta } => {
                        d.float(format!("{dp}/U"), *u);
                        d.tab1(&format!("{dp}/theta"), theta);
                    }
                    EnergyDistribution::WattEnergy { u, a, b } => {
                        d.float(format!("{dp}/U"), *u);
                        d.tab1(&format!("{dp}/a"), a);
                        d.tab1(&format!("{dp}/b"), b);
                    }
                    EnergyDistribution::MadlandNix { efl, efh, t_m } => {
                        d.float(format!("{dp}/EFL"), *efl);
                        d.float(format!("{dp}/EFH"), *efh);
                        d.tab1(&format!("{dp}/T_M"), t_m);
                    }
                }
            }
        }

        Section::Mf6(s) => {
            d.int(format!("{path}/ZA"), s.za);
            d.float(format!("{path}/AWR"), s.awr);
            d.int(format!("{path}/JP"), s.jp);
            d.int(format!("{path}/LCT"), s.lct);
            d.int(format!("{path}/NK"), s.nk);
            for (i, p) in s.products.iter().enumerate() {
                let pp = format!("{path}/products/{i}");
                d.int(format!("{pp}/ZAP"), p.zap);
                d.float(format!("{pp}/AWP"), p.awp);
                d.int(format!("{pp}/LIP"), p.lip);
                d.int(format!("{pp}/LAW"), p.law);
                d.tab1(&format!("{pp}/y_i"), &p.yield_);
                dump_mf6_distribution(d, &format!("{pp}/distribution"), &p.distribution);
            }
        }

        Section::Unparsed { .. } => {}
    }
}

// --------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

/// The golden file, split into the structural records and the value map.
struct Golden {
    source: String,
    n_materials: usize,
    /// (material index, MF, MT) -> body line count.
    sections: BTreeMap<(usize, i32, i32), usize>,
    /// material index -> MAT number.
    mats: BTreeMap<usize, i32>,
    values: BTreeMap<String, Value>,
}

fn parse_golden(text: &str, name: &str) -> Golden {
    let mut g = Golden {
        source: String::new(),
        n_materials: 0,
        sections: BTreeMap::new(),
        mats: BTreeMap::new(),
        values: BTreeMap::new(),
    };

    for (i, line) in text.lines().enumerate() {
        let at = format!("{name}:{}", i + 1);
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or("");
        let f: Vec<&str> = parts.collect();
        match key {
            "SOURCE" => g.source = f[0].to_string(),
            "MATERIALS" => g.n_materials = f[0].parse().unwrap(),
            "MAT" => {
                g.mats.insert(f[0].parse().unwrap(), f[1].parse().unwrap());
            }
            "SECTION" => {
                let k = (
                    f[0].parse().unwrap(),
                    f[1].parse().unwrap(),
                    f[2].parse().unwrap(),
                );
                g.sections.insert(k, f[3].parse().unwrap());
            }
            "V" => {
                let path = f[0].to_string();
                let rest = &f[2..];
                let value = match f[1] {
                    "F" => Value::Floats(
                        rest.iter()
                            .map(|s| {
                                s.parse()
                                    .unwrap_or_else(|_| panic!("{at}: bad float {s:?}"))
                            })
                            .collect(),
                    ),
                    "I" => Value::Ints(
                        rest.iter()
                            .map(|s| s.parse().unwrap_or_else(|_| panic!("{at}: bad int {s:?}")))
                            .collect(),
                    ),
                    "T" => {
                        let hex = rest.first().copied().unwrap_or("");
                        let bytes: Vec<u8> = (0..hex.len())
                            .step_by(2)
                            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                            .collect();
                        Value::Text(String::from_utf8(bytes).unwrap())
                    }
                    other => panic!("{at}: unknown value tag {other:?}"),
                };
                g.values.insert(path, value);
            }
            other => panic!("{at}: unknown golden record {other:?}"),
        }
    }
    g
}

/// Compare one golden file against the Rust reader. Returns paths compared.
fn check(golden_path: &Path) -> usize {
    let text = std::fs::read_to_string(golden_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", golden_path.display()));
    let name = golden_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let g = parse_golden(&text, &name);

    let source_path = repo_root().join(&g.source);
    let endf_text = std::fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", source_path.display()));

    let materials = materials_from_str(&endf_text)
        .unwrap_or_else(|e| panic!("{name}: the Rust reader failed on {}: {e}", g.source));

    assert_eq!(materials.len(), g.n_materials, "{name}: material count");

    let mut d = Dump::default();
    let mut sections: BTreeMap<(usize, i32, i32), usize> = BTreeMap::new();
    for (m, material) in materials.iter().enumerate() {
        assert_eq!(
            Some(&material.mat),
            g.mats.get(&m),
            "{name}: material {m} number"
        );
        for (&(mf, mt), body) in &material.section_text {
            sections.insert((m, mf, mt), body.lines().count());
        }
        for (&(mf, mt), section) in &material.section_data {
            dump_section(&mut d, &format!("{m}/{mf}/{mt}"), section);
        }
    }

    assert_eq!(sections, g.sections, "{name}: section splitting differs");

    // Paths on one side and not the other: a renamed, dropped or added field.
    let ours: BTreeSet<&String> = d.map.keys().collect();
    let theirs: BTreeSet<&String> = g.values.keys().collect();
    let missing: Vec<&&String> = theirs.difference(&ours).take(10).collect();
    let extra: Vec<&&String> = ours.difference(&theirs).take(10).collect();
    assert!(
        missing.is_empty(),
        "{name}: the Rust reader did not produce {} paths, e.g. {missing:?}",
        theirs.difference(&ours).count()
    );
    assert!(
        extra.is_empty(),
        "{name}: the Rust reader produced {} paths the Python reader does not, e.g. {extra:?}",
        ours.difference(&theirs).count()
    );

    for (path, want) in &g.values {
        let got = &d.map[path];
        assert_eq!(
            got.kind(),
            want.kind(),
            "{name}: {path} is {} in Rust and {} in Python",
            got.kind(),
            want.kind()
        );
        match (got, want) {
            // Interpolation is arithmetic, not parsing: the two languages
            // evaluate the same expression but need not round identically once
            // logs and exps are involved.
            (Value::Floats(a), Value::Floats(b)) if path.ends_with("/evaly") => {
                assert_eq!(a.len(), b.len(), "{name}: {path} length");
                for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
                    let tol = EVAL_TOL * y.abs().max(1.0);
                    assert!(
                        (x - y).abs() <= tol,
                        "{name}: {path}[{i}]: rust {x} != python {y}"
                    );
                }
            }
            _ => assert_eq!(got, want, "{name}: {path}"),
        }
    }

    g.values.len()
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
    println!("{} golden files, {total} paths compared", goldens.len());
}

#[test]
fn unported_files_keep_their_text() {
    // The port proceeds file by file, so a section with no Rust parser must
    // still round-trip its text for the Python reader to fall back to.
    //
    // The photo-atomic fixture is used because every file in the neutron ones
    // is now ported; MF=23 and MF=27 are still outstanding here.
    let path = repo_root().join("tests").join("photoat-001_H_000.endf");
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
