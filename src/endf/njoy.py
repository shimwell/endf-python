# SPDX-FileCopyrightText: 2011-2023 OpenMC contributors
# SPDX-FileCopyrightText: 2023-2025 Paul Romano
# SPDX-License-Identifier: MIT
#
# Ported from openmc/data/njoy.py. Reads the evaluation metadata through
# endf.Material rather than openmc's Evaluation, and builds each module's
# substitutions explicitly instead of formatting against locals(). The thermal
# scattering path (make_ace_thermal and its _THERMAL_DATA table) is not ported.

"""Drive NJOY to turn an ENDF evaluation into an ACE library.

ENDF evaluations describe cross sections at 0 K with resonance parameters rather
than pointwise data, so they have to be reconstructed and Doppler broadened
before they can be used. That is NJOY's job: RECONR reconstructs the resonances,
BROADR broadens to temperature, HEATR adds heating numbers, GASPR adds gas
production, PURR builds unresolved resonance probability tables, and ACER writes
the result out in ACE format.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from typing import Iterable, Optional, Union

from .fileutils import PathLike
from .material import Material, _LIBRARY


__all__ = ["run", "make_pendf", "make_ace"]


_TEMPLATE_RECONR = """
reconr / %%%%%%%%%%%%%%%%%%% Reconstruct XS for neutrons %%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf}
'{library} PENDF for {zsymam}'/
{mat} 2/
{error}/ err
'{library}: {zsymam}'/
'Processed by NJOY'/
0/
"""

_TEMPLATE_BROADR = """
broadr / %%%%%%%%%%%%%%%%%%%%%%% Doppler broaden XS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npendf} {nbroadr}
{mat} {num_temp} 0 0 0. /
{error}/ errthn
{temps}
0/
"""

_TEMPLATE_HEATR = """
heatr / %%%%%%%%%%%%%%%%%%%%%%%%% Add heating kerma %%%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr} /
{mat} 4 0 0 0 /
302 318 402 444 /
"""

_TEMPLATE_HEATR_LOCAL = """
heatr / %%%%%%%%%%%%%%%%% Add heating kerma (local photons) %%%%%%%%%%%%%%%%%%%%
{nendf} {nheatr_in} {nheatr_local} /
{mat} 4 0 0 1 /
302 318 402 444 /
"""

_TEMPLATE_GASPR = """
gaspr / %%%%%%%%%%%%%%%%%%%%%%%%% Add gas production %%%%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {ngaspr_in} {ngaspr} /
"""

_TEMPLATE_PURR = """
purr / %%%%%%%%%%%%%%%%%%%%%%%% Add probability tables %%%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {npurr_in} {npurr} /
{mat} {num_temp} 1 20 64 /
{temps}
1.e10
0/
"""

_TEMPLATE_ACER = """
acer / %%%%%%%%%%%%%%%%%%%%%%%% Write out in ACE format %%%%%%%%%%%%%%%%%%%%%%%%
{nendf} {nacer_in} 0 {nace} {ndir}
1 0 1 .{ext} /
'{library}: {zsymam} at {temperature}'/
{mat} {temperature}
1 1 {ismooth}/
/
"""


def run(commands: str, tapein: dict, tapeout: dict,
        input_filename: Optional[PathLike] = None, stdout: bool = False,
        njoy_exec: str = 'njoy'):
    """Run NJOY with the given commands.

    Parameters
    ----------
    commands
        Input commands for NJOY
    tapein
        Mapping of tape numbers to paths for any input files
    tapeout
        Mapping of tape numbers to paths for any output files
    input_filename
        File to write the NJOY input commands to
    stdout
        Whether to display NJOY's output while it runs
    njoy_exec
        Path to the NJOY executable

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns a non-zero status

    """
    if input_filename is not None:
        with open(str(input_filename), 'w') as f:
            f.write(commands)

    with tempfile.TemporaryDirectory() as tmpdir:
        # NJOY refers to its files by unit number, so copy the inputs into
        # place as tape20, tape21 and so on.
        for tape_num, filename in tapein.items():
            tmpfilename = os.path.join(tmpdir, f'tape{tape_num}')
            shutil.copy(str(filename), tmpfilename)

        njoy = Popen([njoy_exec], cwd=tmpdir, stdin=PIPE, stdout=PIPE,
                     stderr=STDOUT, universal_newlines=True)

        njoy.stdin.write(commands)
        njoy.stdin.flush()
        lines = []
        while True:
            line = njoy.stdout.readline()
            if not line and njoy.poll() is not None:
                break
            lines.append(line)
            if stdout:
                print(line, end='')

        if njoy.returncode != 0:
            raise CalledProcessError(njoy.returncode, njoy_exec,
                                     ''.join(lines))

        for tape_num, filename in tapeout.items():
            tmpfilename = os.path.join(tmpdir, f'tape{tape_num}')
            if os.path.isfile(tmpfilename):
                shutil.move(tmpfilename, str(filename))


def make_pendf(filename: PathLike, pendf: PathLike = 'pendf', **kwargs):
    """Generate a pointwise ENDF file from an ENDF file.

    Parameters
    ----------
    filename
        Path to the ENDF file
    pendf
        Path of the pointwise ENDF file to write
    **kwargs
        Keyword arguments passed to :func:`make_ace`. Every NJOY module other
        than pendf defaults to False.

    Raises
    ------
    subprocess.CalledProcessError
        If the NJOY process returns a non-zero status

    """
    for key in ('broadr', 'heatr', 'gaspr', 'purr', 'acer'):
        kwargs.setdefault(key, False)
    make_ace(filename, pendf=pendf, **kwargs)


def make_ace(
    filename: PathLike,
    temperatures: Optional[Iterable[float]] = None,
    acer: Union[bool, PathLike] = True,
    xsdir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    pendf: Union[bool, PathLike] = False,
    error: float = 0.001,
    broadr: Union[bool, PathLike] = True,
    heatr: Union[bool, PathLike] = True,
    gaspr: Union[bool, PathLike] = True,
    purr: Union[bool, PathLike] = True,
    material: Optional[Material] = None,
    smoothing: bool = True,
    **kwargs,
):
    """Generate an ACE file from an ENDF file.

    Parameters
    ----------
    filename
        Path to the ENDF file
    temperatures
        Temperatures in Kelvin to produce data at. Defaults to room temperature
        (293.6 K).
    acer, pendf, broadr, heatr, gaspr, purr
        Whether to run the corresponding NJOY module. A path may be given
        instead of True to control where that module's output tape is written;
        otherwise the file is named after the module and placed in
        ``output_dir``.
    xsdir
        Path of the xsdir file to write. Defaults to sitting alongside ``acer``.
    output_dir
        Directory to write output files to. Defaults to the current directory.
    error
        Fractional error tolerance for the NJOY reconstruction and broadening
    material
        Material to use, when the ENDF file holds more than one evaluation
    smoothing
        Whether ACER applies its thinning and smoothing to the elastic and
        capture cross sections at low energy
    **kwargs
        Keyword arguments passed to :func:`run`

    Raises
    ------
    IOError
        If ``output_dir`` is not a directory
    subprocess.CalledProcessError
        If the NJOY process returns a non-zero status

    """
    if output_dir is None:
        output_dir = Path()
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise IOError(f"{output_dir} is not a directory")

    mat_obj = material if material is not None else Material(filename)
    metadata = mat_obj.section_data[1, 451]
    mat = mat_obj.MAT
    zsymam = metadata['ZSYMAM']
    isomeric_state = metadata['LISO']

    library = '{}-{}.{}'.format(
        _LIBRARY.get(metadata['NLIB'], 'Unknown'),
        metadata['NVER'], metadata['LREL'])

    if temperatures is None:
        temperatures = [293.6]
    temperatures = list(temperatures)
    num_temp = len(temperatures)
    temps = ' '.join(str(i) for i in temperatures)

    # NJOY refers to files by unit number. Each module reads the tape the
    # previous one wrote, so the numbers are threaded through in sequence.
    commands = ""
    nendf, npendf = 20, 21
    tapein = {nendf: filename}
    tapeout = {}
    subs = {'nendf': nendf, 'npendf': npendf, 'mat': mat, 'error': error,
            'library': library, 'zsymam': zsymam, 'num_temp': num_temp,
            'temps': temps}

    if pendf:
        tapeout[npendf] = (output_dir / "pendf") if pendf is True else pendf

    commands += _TEMPLATE_RECONR
    nlast = npendf

    if broadr:
        nbroadr = nlast + 1
        tapeout[nbroadr] = (output_dir / "broadr") if broadr is True else broadr
        subs['nbroadr'] = nbroadr
        commands += _TEMPLATE_BROADR
        nlast = nbroadr

    if heatr:
        # Two HEATR runs: one assuming photons deposit their energy locally and
        # one assuming they carry it away.
        nheatr_in = nlast
        nheatr_local = nheatr_in + 1
        tapeout[nheatr_local] = (output_dir / "heatr_local") \
            if heatr is True else str(heatr) + '_local'
        subs['nheatr_in'] = nheatr_in
        subs['nheatr_local'] = nheatr_local
        commands += _TEMPLATE_HEATR_LOCAL

        nheatr = nheatr_local + 1
        tapeout[nheatr] = (output_dir / "heatr") if heatr is True else heatr
        subs['nheatr'] = nheatr
        commands += _TEMPLATE_HEATR
        nlast = nheatr

    if gaspr:
        ngaspr_in = nlast
        ngaspr = ngaspr_in + 1
        tapeout[ngaspr] = (output_dir / "gaspr") if gaspr is True else gaspr
        subs['ngaspr_in'] = ngaspr_in
        subs['ngaspr'] = ngaspr
        commands += _TEMPLATE_GASPR
        nlast = ngaspr

    if purr:
        npurr_in = nlast
        npurr = npurr_in + 1
        tapeout[npurr] = (output_dir / "purr") if purr is True else purr
        subs['npurr_in'] = npurr_in
        subs['npurr'] = npurr
        commands += _TEMPLATE_PURR
        nlast = npurr

    commands = commands.format(**subs)

    if acer:
        nacer_in = nlast
        for i, temperature in enumerate(temperatures):
            # One ACER run per temperature, each writing its own ACE and xsdir
            nace = nacer_in + 1 + 2*i
            ndir = nace + 1
            commands += _TEMPLATE_ACER.format(
                nendf=nendf, nacer_in=nacer_in, nace=nace, ndir=ndir,
                ext=f'{i + 1:02}', library=library, zsymam=zsymam,
                temperature=temperature, mat=mat, ismooth=int(smoothing))
            tapeout[nace] = output_dir / f"ace_{temperature:.1f}"
            tapeout[ndir] = output_dir / f"xsdir_{temperature:.1f}"

    commands += 'stop\n'
    run(commands, tapein, tapeout, **kwargs)

    if acer:
        ace = (output_dir / "ace") if acer is True else Path(acer)
        xsdir = (ace.parent / "xsdir") if xsdir is None else Path(xsdir)
        with ace.open('w') as ace_file, xsdir.open('w') as xsdir_file:
            for temperature in temperatures:
                text = (output_dir / f"ace_{temperature:.1f}").read_text()

                # A metastable target is not encoded in the ZAID that ACER
                # writes, so add 400 to it the way MCNP libraries do.
                if isomeric_state > 0:
                    mass_first_digit = int(text[3])
                    if mass_first_digit <= 2:
                        text = text[:3] + str(mass_first_digit + 4) + text[4:]

                ace_file.write(text)
                xsdir_file.write(
                    (output_dir / f"xsdir_{temperature:.1f}").read_text())

        for temperature in temperatures:
            (output_dir / f"ace_{temperature:.1f}").unlink()
            (output_dir / f"xsdir_{temperature:.1f}").unlink()
