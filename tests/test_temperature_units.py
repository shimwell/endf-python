"""Pin the temperature units at each boundary.

Temperature is carried in three different units between an ACE file and an
IncidentNeutron, which is easy to get wrong: OpenMC's ``ace.Table.temperature``
returns the raw kT in MeV despite the name, while this library's returns Kelvin.
These tests fix the meaning of each attribute so a future change has to be
deliberate.
"""

import math

import pytest

from endf.data import K_BOLTZMANN, EV_PER_MEV


# Room temperature as ACE files store it: kT in MeV.
ROOM_KT_MEV = 2.53e-8
ROOM_KELVIN = 293.6


def test_boltzmann_constant_is_in_ev_per_kelvin():
    # 8.617333262e-5 eV/K, the CODATA value.
    assert K_BOLTZMANN == pytest.approx(8.617333262e-5)


def test_kt_in_mev_converts_to_room_temperature():
    """The chain used everywhere: kT [MeV] -> eV -> Kelvin."""
    kelvin = ROOM_KT_MEV * EV_PER_MEV / K_BOLTZMANN
    assert kelvin == pytest.approx(ROOM_KELVIN, abs=0.5)


def test_kt_in_ev_is_the_familiar_room_temperature_value():
    """IncidentNeutron.kTs is in eV, where room temperature is 0.0253."""
    assert ROOM_KT_MEV * EV_PER_MEV == pytest.approx(0.0253)


def test_temperature_str_rounds_to_whole_kelvin():
    from endf.data import temperature_str
    assert temperature_str(ROOM_KELVIN) == '294K'
    assert temperature_str(900.0000074) == '900K'
    assert temperature_str(0.0) == '0K'


@pytest.mark.parametrize('kelvin', [250.0, 293.6, 600.0, 900.0, 1200.0, 2500.0])
def test_round_trip_kelvin_through_kt(kelvin):
    kt_mev = kelvin * K_BOLTZMANN / EV_PER_MEV
    assert kt_mev * EV_PER_MEV / K_BOLTZMANN == pytest.approx(kelvin)


def test_ace_table_exposes_both_units():
    """Table.kT is the raw MeV value from the file; Table.temperature is Kelvin.

    Built directly rather than from a file, since the units are a property of the
    class and not of any particular evaluation.
    """
    from endf.ace import Table
    table = Table('26056.01c', 55.45443, ROOM_KT_MEV, None, [0] * 17,
                  [0] * 33, None)
    assert table.kT == ROOM_KT_MEV
    assert table.temperature == pytest.approx(ROOM_KELVIN, abs=0.5)
    # The two must not be confused: they differ by more than ten orders of
    # magnitude, so a mix-up is never a small numerical error.
    assert math.log10(table.temperature / table.kT) > 9
