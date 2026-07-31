# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/data/fission_energy.py, reading through endf.Material.
# The raw MF=1/MT=458 section text is parsed here rather than reusing the
# already-parsed section_data, because the components are read as strided slices
# of one LIST record and that layout has to be preserved exactly.

"""Energy release from fission, by component."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO

from numpy.polynomial import Polynomial

from .data import EV_PER_MEV
from .function import Tabulated1D
from .material import Material
from .records import get_cont_record, get_list_record, get_tab1_record


__all__ = ["FissionEnergyRelease"]


# Components of the fission energy release, in the order they appear in the
# MF=1/MT=458 LIST record.
_NAMES = (
    'fragments', 'prompt_neutrons', 'delayed_neutrons',
    'prompt_photons', 'delayed_photons', 'betas',
    'neutrinos', 'recoverable', 'total'
)


class FissionEnergyRelease:
    """Energy released by fission, split into its components.

    Each component is a callable giving the energy released in [eV] as a
    function of the incident neutron energy in [eV].

    Parameters
    ----------
    fragments
        Kinetic energy of the fission fragments
    prompt_neutrons
        Kinetic energy of the prompt fission neutrons
    delayed_neutrons
        Kinetic energy of the delayed fission neutrons
    prompt_photons
        Energy of the prompt fission photons
    delayed_photons
        Energy of the delayed fission photons
    betas
        Energy of the delayed beta particles
    neutrinos
        Energy of the neutrinos

    Attributes
    ----------
    fragments, prompt_neutrons, delayed_neutrons, prompt_photons, \
    delayed_photons, betas, neutrinos : Callable
        The components listed above.
    recoverable : Callable
        Energy release that can be recovered in a reactor, i.e. everything
        except the neutrinos.
    total : Callable
        Total energy release, including the neutrinos.

    """

    def __init__(self, fragments, prompt_neutrons, delayed_neutrons,
                 prompt_photons, delayed_photons, betas, neutrinos):
        self.fragments = fragments
        self.prompt_neutrons = prompt_neutrons
        self.delayed_neutrons = delayed_neutrons
        self.prompt_photons = prompt_photons
        self.delayed_photons = delayed_photons
        self.betas = betas
        self.neutrinos = neutrinos

    def __repr__(self) -> str:
        return "<FissionEnergyRelease>"

    @property
    def recoverable(self):
        components = ['fragments', 'prompt_neutrons', 'delayed_neutrons',
                      'prompt_photons', 'delayed_photons', 'betas']
        return lambda E: sum(getattr(self, c)(E) for c in components)

    @property
    def total(self):
        components = ['fragments', 'prompt_neutrons', 'delayed_neutrons',
                      'prompt_photons', 'delayed_photons', 'betas',
                      'neutrinos']
        return lambda E: sum(getattr(self, c)(E) for c in components)

    @property
    def q_prompt(self):
        """Prompt fission Q value, i.e. the prompt energy release less the
        incident neutron energy."""
        return lambda E: (
            self.fragments(E) + self.prompt_neutrons(E)
            + self.prompt_photons(E) - E)

    @property
    def q_recoverable(self):
        return lambda E: self.recoverable(E) - E

    @property
    def q_total(self):
        return lambda E: self.total(E) - E

    @classmethod
    def from_endf(cls, material_or_filename, incident_neutron
                  ) -> FissionEnergyRelease:
        """Read fission energy release data from an ENDF evaluation.

        Parameters
        ----------
        material_or_filename : endf.Material or path-like
            ENDF evaluation to read from
        incident_neutron : endf.IncidentNeutron
            Corresponding incident neutron data, needed for the nu values that
            the Sher-Beck energy dependence uses

        Returns
        -------
        Fission energy release data

        """
        if isinstance(material_or_filename, Material):
            material = material_or_filename
        else:
            material = Material(material_or_filename)

        metadata = material.section_data[1, 451]
        Z, A = divmod(metadata['ZA'], 1000)
        if Z != incident_neutron.atomic_number:
            raise ValueError("The atomic number of the ENDF evaluation does "
                             "not match the given IncidentNeutron.")
        if A != incident_neutron.mass_number:
            raise ValueError("The atomic mass of the ENDF evaluation does not "
                             "match the given IncidentNeutron.")
        if metadata['LISO'] != incident_neutron.metastable:
            raise ValueError("The metastable state of the ENDF evaluation does "
                             "not match the given IncidentNeutron.")
        if metadata['LFI'] != 1:
            raise ValueError("The ENDF evaluation is not fissionable.")

        if (1, 458) not in material.section_text:
            raise ValueError("ENDF evaluation does not have MF=1, MT=458.")

        file_obj = StringIO(material.section_text[1, 458])

        # Whether any components are given as tabulated functions
        items = get_cont_record(file_obj)
        lfc = items[3]
        nfc = items[5]

        items, data = get_list_record(file_obj)
        npoly = items[3]

        functions = {}
        for i, name in enumerate(_NAMES):
            # Each component occupies one of 18 slots, values interleaved with
            # their uncertainties
            coeffs = data[2*i::18]

            # recoverable and total are recomputed from the components
            if name in ('recoverable', 'total'):
                continue

            # In ENDF/B-VII.1 the second-order coefficients were mistakenly left
            # in MeV. Detect that and fix it: a 5 MeV neutron causing a change of
            # more than 100 MeV cannot be right.
            if npoly == 2:
                if abs(coeffs[2]) * (5e6)**2 > 1e8:
                    coeffs[2] /= EV_PER_MEV

            if npoly > 0:
                functions[name] = Polynomial(coeffs)
                continue

            # Only a single coefficient, so the energy dependence comes from the
            # Sher-Beck formula
            zeroth_order = coeffs[0]
            if name in ('delayed_photons', 'betas'):
                func = Polynomial((zeroth_order, -0.075))
            elif name == 'neutrinos':
                func = Polynomial((zeroth_order, -0.105))
            elif name == 'prompt_neutrons':
                # Needs nu. ENDF-102 does not say whether the prompt or total
                # value should be used, but the delayed fraction is small enough
                # that it makes no practical difference. MT=18 may be absent, in
                # which case MT=19 is tried.
                if 18 in incident_neutron and not incident_neutron[18].redundant:
                    fission_rx = incident_neutron[18]
                elif 19 in incident_neutron:
                    fission_rx = incident_neutron[19]
                else:
                    raise ValueError(
                        "IncidentNeutron data has no fission reaction.")

                nu = [p.yield_ for p in fission_rx.products
                      if p.name == 'neutron'
                      and p.emission_mode in ('prompt', 'total')]
                if len(nu) == 0:
                    raise ValueError(
                        "Nu data is needed to compute fission energy release "
                        "with the Sher-Beck format.")
                if len(nu) > 1:
                    raise ValueError("Ambiguous prompt/total nu value.")

                nu = nu[0]
                if isinstance(nu, Tabulated1D):
                    func = deepcopy(nu)
                    func.y = (zeroth_order + 1.307*nu.x
                              - 8.07e6*(nu.y - nu.y[0]))
                elif isinstance(nu, Polynomial):
                    if len(nu) == 1:
                        func = Polynomial([zeroth_order, 1.307])
                    else:
                        func = Polynomial(
                            [zeroth_order, 1.307 - 8.07e6*nu.coef[1]]
                            + [-8.07e6*c for c in nu.coef[2:]])
            else:
                func = Polynomial(coeffs)

            functions[name] = func

        # Tabulated components override the polynomial forms
        if lfc == 1:
            for _ in range(nfc):
                items, eifc = get_tab1_record(file_obj)
                functions[_NAMES[items[3] - 1]] = eifc

        return cls(**functions)
