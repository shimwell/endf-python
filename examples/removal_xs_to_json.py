"""Extract neutron and photon removal cross sections for Fe and write to JSON.

Outputs two JSON files, each containing energy (eV) and removal cross section
(barns) arrays ready for use in kernel density estimation or other analysis.

Usage:
    python examples/removal_xs_to_json.py

Adjust NEUTRON_ENDF and PHOTON_ENDF paths below to point at your ENDF files.
"""

import json
import pathlib

import endf

# --- file paths (change these to point at your ENDF/B-VIII.1 files) --------
NEUTRON_ENDF = pathlib.Path.home() / "nuclear_data/endfb-viii.0-endf/neutron/n-026_Fe_056.endf"
PHOTON_ENDF  = pathlib.Path.home() / "nuclear_data/endfb-viii.0-endf/photon/photoat-026_Fe_000.endf"

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent

# --- neutron removal cross section for Fe-56 --------------------------------
#
# mu_cutoff controls which elastically scattered neutrons count as "removed".
# It is the cosine of the scattering angle that separates the forward cone
# (particles considered to keep going) from the rest (particles considered
# removed).
#
#   sigma_r(E) = sigma_total(E) - f_forward(E) * sigma_elastic(E)
#
# where f_forward is the fraction of elastic scattering into [mu_cutoff, 1].
#
#   mu_cutoff =  0.0  ->  forward hemisphere (theta < 90 deg) keeps going,
#                         backscatter is removed.  This is the most common
#                         choice for shielding / removal cross sections.
#   mu_cutoff = -1.0  ->  ALL elastic is "forward", so removal = total - elastic
#   mu_cutoff =  1.0  ->  NO  elastic is "forward", so removal = total
#
# For a kernel-density application you most likely want the default (0.0).
neutron_data = endf.IncidentNeutron.from_endf(NEUTRON_ENDF)
neutron_removal = neutron_data.removal_xs(temperature="0K", mu_cutoff=0.0)

neutron_out = {
    "description": f"Neutron removal cross section for {neutron_data.name}",
    "energy_eV": neutron_removal.x.tolist(),
    "removal_xs_barns": neutron_removal.y.tolist(),
}

neutron_file = OUTPUT_DIR / "neutron_removal_xs_Fe56.json"
neutron_file.write_text(json.dumps(neutron_out, indent=2))
print(f"Wrote {len(neutron_removal.x)} points to {neutron_file}")

# --- photon removal cross section for Fe ------------------------------------
#
# For photons the same mu_cutoff logic applies, but the forward-scattered
# component is coherent (Rayleigh) scattering rather than elastic neutron
# scattering:
#
#   sigma_r(E) = sigma_total(E) - f_forward(E) * sigma_coherent(E)
#
# f_forward is derived from the Thomson-weighted atomic form factor.
# mu_cutoff=0.0 again means the forward hemisphere is not "removed".
photon_data = endf.IncidentPhoton.from_endf(PHOTON_ENDF)
photon_removal = photon_data.removal_xs(mu_cutoff=0.0)

photon_out = {
    "description": f"Photon removal cross section for {photon_data.name}",
    "energy_eV": photon_removal.x.tolist(),
    "removal_xs_barns": photon_removal.y.tolist(),
}

photon_file = OUTPUT_DIR / "photon_removal_xs_Fe.json"
photon_file.write_text(json.dumps(photon_out, indent=2))
print(f"Wrote {len(photon_removal.x)} points to {photon_file}")
