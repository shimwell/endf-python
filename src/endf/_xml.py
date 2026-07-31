# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-License-Identifier: MIT
#
# Ported from openmc/_xml.py


def clean_indentation(element, level=0, spaces_per_level=2):
    """Walk an ElementTree and add indentation so it pretty-prints.

    Adapted from https://effbot.org/zone/element-lib.htm#prettyprint
    """
    i = "\n" + level*spaces_per_level*" "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + spaces_per_level*" "
        if not element.tail or not element.tail.strip():
            element.tail = i
        for sub_element in element:
            clean_indentation(sub_element, level+1, spaces_per_level)
        if not sub_element.tail or not sub_element.tail.strip():
            sub_element.tail = i
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i


def get_text(elem, name, default=None):
    """Retrieve text of an attribute or subelement."""
    if name in elem.attrib:
        return elem.get(name, default)
    child = elem.find(name)
    return child.text if child is not None else default


def get_elem_list(elem, name, dtype=int):
    """Retrieve a whitespace-separated list of values from an attribute or
    subelement, each converted with ``dtype``. Returns None if absent."""
    text = get_text(elem, name)
    if text is not None:
        return [dtype(x) for x in text.split()]
