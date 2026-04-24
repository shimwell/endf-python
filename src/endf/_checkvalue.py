# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-License-Identifier: MIT
#
# Minimal subset of openmc/checkvalue.py, ported for use inside endf-python.


def check_type(name, value, expected_type, expected_iter_type=None):
    """Ensure ``value`` is an instance of ``expected_type``.

    If ``expected_iter_type`` is given, each element of ``value`` must also
    be an instance of that type.
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Unable to set {name!r} to {value!r} which is not of type "
            f"{expected_type}"
        )

    if expected_iter_type:
        for item in value:
            if not isinstance(item, expected_iter_type):
                raise TypeError(
                    f"Unable to set {name!r} to {value!r}; element "
                    f"{item!r} is not of type {expected_iter_type}"
                )


def check_greater_than(name, value, minimum, equality=False):
    """Ensure ``value`` is greater than ``minimum``."""
    if equality:
        if value < minimum:
            raise ValueError(
                f"{name} must be greater than or equal to {minimum}"
            )
    else:
        if value <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
