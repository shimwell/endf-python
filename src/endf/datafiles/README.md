# Auxiliary photon data

These files are not ENDF evaluations. They hold data needed to describe photon
and electron interactions that the photoatomic sublibrary does not carry, and
they are attached to an `IncidentPhoton` by `_add_compton_profiles` and
`_add_bremsstrahlung`.

All three were copied verbatim from OpenMC's `openmc/data/` directory, which is
MIT licensed. The underlying evaluations are older than OpenMC and are credited
below.

| File | Contents | Source |
| --- | --- | --- |
| `compton_profiles.h5` | Hartree-Fock Compton profiles J(pz) per subshell for Z = 1 to 100 | Biggs, Mendelsohn and Mann, *Atomic Data and Nuclear Data Tables* **16** (1975) 201 |
| `BREMX.DAT` | Scaled bremsstrahlung differential cross sections, 57 electron energies by 30 reduced photon energies, for Z = 1 to 100 | Seltzer and Berger, *Atomic Data and Nuclear Data Tables* **35** (1986) 345 |
| `density_effect.h5` | Mean excitation energies and subshell ionization energies for the density effect correction, Z = 1 to 100 | Sternheimer, Berger and Seltzer, *Atomic Data and Nuclear Data Tables* **30** (1984) 261 |

## Known follow-up

The two HDF5 files are the only reason this package needs `h5py`, and the cubic
spline resampling of `BREMX.DAT` is the only reason it needs `scipy`. Both are
loaded once and cached, and neither format is load bearing: converting them to
`.npz` and either precomputing or reimplementing the spline would drop both
dependencies. Deferred deliberately, see
fusion-neutronics/nuclear_data_to_yamc_format#19.
