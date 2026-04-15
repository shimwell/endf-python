"""Extract neutron-induced gamma production cross sections and write to JSON.

Outputs a JSON file containing energy (eV) and total gamma production cross
section (barns) arrays.  This represents the total rate at which photons are
produced per unit neutron flux — summed over all reactions ((n,gamma),
(n,n'gamma), etc.) — and is useful as a secondary-gamma source term in
point kernel shielding calculations.

Usage:
    python examples/gamma_production_xs_to_json.py

Adjust NEUTRON_ENDF below to point at your ENDF file.
"""

import json
import pathlib

import endf

# --- file path (change this to point at your ENDF/B-VIII.1 file) ------------
NEUTRON_ENDF = (
    pathlib.Path.home()
    / "nuclear_data/endfb-viii.0-endf/neutron/n-026_Fe_056.endf"
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent

# --- gamma production cross section for Fe-56 --------------------------------
#
# The total gamma production cross section is:
#
#   sigma_gamma(E) = SUM_over_MT [ Y_MT(E) * sigma_MT(E) ]   (MF=12 reactions)
#                  + SUM_over_MT [ sigma_gamma_MT(E) ]         (MF=13 reactions)
#
# where Y_MT is the photon multiplicity (photons per reaction) from MF=12
# and sigma_gamma_MT is the photon production cross section from MF=13.
#
# This gives the total number of photons produced per neutron collision,
# weighted by the reaction probability.  In a point kernel code you would
# use this as the volumetric source strength for secondary gammas at each
# point in the shield, then attenuate each gamma energy to the detector
# using the photon removal cross section.

neutron_data = endf.IncidentNeutron.from_endf(NEUTRON_ENDF)
gamma_xs = neutron_data.gamma_production_xs(temperature="0K")

output = {
    "description": f"Total gamma production cross section for {neutron_data.name}",
    "energy_eV": gamma_xs.x.tolist(),
    "gamma_production_xs_barns": gamma_xs.y.tolist(),
}

out_file = OUTPUT_DIR / "gamma_production_xs_Fe56.json"
out_file.write_text(json.dumps(output, indent=2))
print(f"Wrote {len(gamma_xs.x)} points to {out_file}")
