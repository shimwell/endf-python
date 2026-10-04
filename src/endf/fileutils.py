# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT

import io
import lzma
import os
from typing import Union

# Type for arguments that accept file paths
PathLike = Union[str, bytes, os.PathLike]

#: Suffix marking a file compressed with :mod:`lzma`.
XZ_SUFFIX = '.xz'


def is_compressed(filename: PathLike) -> bool:
    """Whether a path names an xz-compressed file."""
    return str(filename).endswith(XZ_SUFFIX)


def open_text(filename: PathLike, encoding=None):
    """Open a possibly-compressed ENDF or ACE file for reading as text.

    Evaluations are large and highly repetitive, so keeping them compressed
    costs a little CPU and saves most of the disk. A path ending in ``.xz`` is
    decompressed on the fly; anything else is opened as it always was.
    """
    if is_compressed(filename):
        # Decompressed whole rather than streamed: the readers seek back and
        # forth to find material boundaries, and seeking inside a compressed
        # stream restarts the decoder. An evaluation is a couple of megabytes.
        with lzma.open(str(filename), 'rt', encoding=encoding) as fh:
            return io.StringIO(fh.read())
    return open(str(filename), 'r', encoding=encoding)


def open_binary(filename: PathLike):
    """The same, for a file read as bytes."""
    if is_compressed(filename):
        with lzma.open(str(filename), 'rb') as fh:
            return io.BytesIO(fh.read())
    return open(str(filename), 'rb')
