# SPDX-FileCopyrightText: 2023-2025 OpenMC contributors and Paul Romano
# SPDX-License-Identifier: MIT

from collections.abc import Iterable
from math import exp, log

import numpy as np

from .data import EV_PER_MEV


# ENDF interpolation law codes (the values stored in a TAB1 record's
# interpolation array) mapped to the names used for them elsewhere. The names
# describe how y varies with x, so 'linear-log' is law 3, where y is linear in
# ln(x).
INTERPOLATION_SCHEME = {
    1: 'histogram',
    2: 'linear-linear',
    3: 'linear-log',
    4: 'log-linear',
    5: 'log-log',
}


class Tabulated1D:
    """A one-dimensional tabulated function.

    This class mirrors the TAB1 type from the ENDF-6 format. A tabulated
    function is specified by tabulated (x,y) pairs along with interpolation
    rules that determine the values between tabulated pairs.

    Once an object has been created, it can be used as though it were an actual
    function, e.g.:

    >>> f = Tabulated1D([0, 10], [4, 5])
    >>> [f(xi) for xi in numpy.linspace(0, 10, 5)]
    [4.0, 4.25, 4.5, 4.75, 5.0]

    Parameters
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).

    Attributes
    ----------
    x : Iterable of float
        Independent variable
    y : Iterable of float
        Dependent variable
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).
    n_regions : int
        Number of interpolation regions
    n_pairs : int
        Number of tabulated (x,y) pairs

    """

    def __init__(self, x, y, breakpoints=None, interpolation=None):
        if breakpoints is None or interpolation is None:
            # Single linear-linear interpolation region by default
            self.breakpoints = np.array([len(x)])
            self.interpolation = np.array([2])
        else:
            self.breakpoints = np.asarray(breakpoints, dtype=int)
            self.interpolation = np.asarray(interpolation, dtype=int)

        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def __repr__(self):
        return f"<Tabulated1D: {self.x.size} points, {self.breakpoints.size} regions>"

    def __call__(self, x):
        # Check if input is scalar
        if not isinstance(x, Iterable):
            return self._interpolate_scalar(x)

        x = np.array(x)

        # Create output array
        y = np.zeros_like(x)

        # Get indices for interpolation
        idx = np.searchsorted(self.x, x, side='right') - 1

        # Loop over interpolation regions
        for k in range(len(self.breakpoints)):
            # Get indices for the begining and ending of this region
            i_begin = self.breakpoints[k-1] - 1 if k > 0 else 0
            i_end = self.breakpoints[k] - 1

            # Figure out which idx values lie within this region
            contained = (idx >= i_begin) & (idx < i_end)

            xk = x[contained]                 # x values in this region
            xi = self.x[idx[contained]]       # low edge of corresponding bins
            xi1 = self.x[idx[contained] + 1]  # high edge of corresponding bins
            yi = self.y[idx[contained]]
            yi1 = self.y[idx[contained] + 1]

            if self.interpolation[k] == 1:
                # Histogram
                y[contained] = yi

            elif self.interpolation[k] == 2:
                # Linear-linear
                y[contained] = yi + (xk - xi)/(xi1 - xi)*(yi1 - yi)

            elif self.interpolation[k] == 3:
                # Linear-log
                y[contained] = yi + np.log(xk/xi)/np.log(xi1/xi)*(yi1 - yi)

            elif self.interpolation[k] == 4:
                # Log-linear
                y[contained] = yi*np.exp((xk - xi)/(xi1 - xi)*np.log(yi1/yi))

            elif self.interpolation[k] == 5:
                # Log-log
                y[contained] = (yi*np.exp(np.log(xk/xi)/np.log(xi1/xi)
                                *np.log(yi1/yi)))

        # In some cases, x values might be outside the tabulated region due only
        # to precision, so we check if they're close and set them equal if so.
        y[np.isclose(x, self.x[0], atol=1e-14)] = self.y[0]
        y[np.isclose(x, self.x[-1], atol=1e-14)] = self.y[-1]

        return y

    def _interpolate_scalar(self, x):
        if x <= self._x[0]:
            return self._y[0]
        elif x >= self._x[-1]:
            return self._y[-1]

        # Get the index for interpolation
        idx = np.searchsorted(self._x, x, side='right') - 1

        # Loop over interpolation regions
        for b, p in zip(self.breakpoints, self.interpolation):
            if idx < b - 1:
                break

        xi = self._x[idx]       # low edge of the corresponding bin
        xi1 = self._x[idx + 1]  # high edge of the corresponding bin
        yi = self._y[idx]
        yi1 = self._y[idx + 1]

        if p == 1:
            # Histogram
            return yi

        elif p == 2:
            # Linear-linear
            return yi + (x - xi)/(xi1 - xi)*(yi1 - yi)

        elif p == 3:
            # Linear-log
            return yi + log(x/xi)/log(xi1/xi)*(yi1 - yi)

        elif p == 4:
            # Log-linear
            return yi*exp((x - xi)/(xi1 - xi)*log(yi1/yi))

        elif p == 5:
            # Log-log
            return yi*exp(log(x/xi)/log(xi1/xi)*log(yi1/yi))

    def __len__(self):
        return len(self.x)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def breakpoints(self):
        return self._breakpoints

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def n_pairs(self):
        return len(self.x)

    @property
    def n_regions(self):
        return len(self.breakpoints)

    @x.setter
    def x(self, x):
        self._x = x

    @y.setter
    def y(self, y):
        self._y = y

    @breakpoints.setter
    def breakpoints(self, breakpoints):
        self._breakpoints = breakpoints

    @interpolation.setter
    def interpolation(self, interpolation):
        self._interpolation = interpolation

    @property
    def is_linear(self) -> bool:
        """Whether every interpolation region is lin-lin (ENDF law 2)."""
        return bool(np.all(np.asarray(self.interpolation) == 2))

    def linearize(self, rel_tol: float = 1e-3) -> 'Tabulated1D':
        """Return an equivalent function using only lin-lin interpolation.

        Consumers that treat tabulated pairs as piecewise linear will silently
        change the shape of any region declared with another interpolation law,
        so those regions are resampled densely enough that linear interpolation
        reproduces the declared law.

        Law 2 (lin-lin) regions pass through unchanged, so an already-linear
        function is returned with the same points. Law 1 (histogram) is exact:
        each step becomes a duplicated breakpoint carrying the jump. The smooth
        laws (3 lin-log, 4 log-lin, 5 log-log) are adaptively bisected until
        linear interpolation matches within ``rel_tol``, with a depth cap.
        Intervals where a law is undefined, for instance non-positive values
        under law 4 or 5, fall back to the stored pair.

        Parameters
        ----------
        rel_tol
            Relative tolerance for the bisection of smooth laws

        Returns
        -------
        A function with a single lin-lin interpolation region

        """
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        breakpoints = np.asarray(self.breakpoints, dtype=np.int64)
        interpolation = np.asarray(self.interpolation, dtype=np.int64)

        if self.is_linear:
            return Tabulated1D(x, y)

        def law_value(law, x0, y0, x1, y1, xm):
            """Evaluate the ENDF interpolation law on one interval at xm."""
            if law == 1:
                return y0
            if law == 3:  # y linear in ln(x)
                return y0 + np.log(xm / x0) / np.log(x1 / x0) * (y1 - y0)
            if law == 4:  # ln(y) linear in x
                return y0 * np.exp((xm - x0) / (x1 - x0) * np.log(y1 / y0))
            if law == 5:  # ln(y) linear in ln(x)
                return y0 * np.exp(np.log(xm / x0) / np.log(x1 / x0)
                                   * np.log(y1 / y0))
            return y0 + (xm - x0) / (x1 - x0) * (y1 - y0)

        def law_defined(law, x0, y0, x1, y1):
            if law in (3, 5) and (x0 <= 0.0 or x1 <= 0.0):
                return False
            if law in (4, 5) and (y0 <= 0.0 or y1 <= 0.0):
                return False
            return True

        out_x, out_y = [], []

        def emit(px, py):
            # Skip exact repeats of the previous pair (shared region
            # boundaries), but keep intentional duplicate-x jump pairs.
            if out_x and out_x[-1] == px and out_y[-1] == py:
                return
            out_x.append(px)
            out_y.append(py)

        max_depth = 24
        for k, law in enumerate(interpolation):
            i_begin = int(breakpoints[k - 1]) - 1 if k > 0 else 0
            i_end = int(breakpoints[k]) - 1
            for i in range(i_begin, i_end):
                x0, y0, x1, y1 = x[i], y[i], x[i + 1], y[i + 1]
                emit(x0, y0)
                if x1 <= x0 or law == 2:
                    continue
                if law == 1:
                    # Exact step: hold y0 up to x1, jump there.
                    emit(x1, y0)
                    continue
                if not law_defined(law, x0, y0, x1, y1):
                    continue
                # Adaptive bisection until lin-lin matches the law.
                stack = [(x0, y0, x1, y1, 0)]
                while stack:
                    a, fa, b, fb, depth = stack.pop()
                    m = 0.5 * (a + b)
                    if m <= a or m >= b or depth >= max_depth:
                        emit(b, fb)
                        continue
                    fm = law_value(law, x0, y0, x1, y1, m)
                    f_lin = 0.5 * (fa + fb)
                    if abs(fm - f_lin) <= rel_tol * max(abs(fm), abs(f_lin)):
                        emit(b, fb)
                        continue
                    # Depth-first, left interval first: push right first.
                    stack.append((m, fm, b, fb, depth + 1))
                    stack.append((a, fa, m, fm, depth + 1))
            emit(x[i_end], y[i_end])

        return Tabulated1D(np.asarray(out_x), np.asarray(out_y))

    def integral(self):
        """Integral of the tabulated function over its tabulated range.

        Returns
        -------
        numpy.ndarray
            Array of same length as the tabulated data that represents partial
            integrals from the bottom of the range to each tabulated point.

        """

        # Create output array
        partial_sum = np.zeros(len(self.x) - 1)

        i_low = 0
        for k in range(len(self.breakpoints)):
            # Determine which x values are within this interpolation range
            i_high = self.breakpoints[k] - 1

            # Get x values and bounding (x,y) pairs
            x0 = self.x[i_low:i_high]
            x1 = self.x[i_low + 1:i_high + 1]
            y0 = self.y[i_low:i_high]
            y1 = self.y[i_low + 1:i_high + 1]

            if self.interpolation[k] == 1:
                # Histogram
                partial_sum[i_low:i_high] = y0*(x1 - x0)

            elif self.interpolation[k] == 2:
                # Linear-linear
                m = (y1 - y0)/(x1 - x0)
                partial_sum[i_low:i_high] = (y0 - m*x0)*(x1 - x0) + \
                                            m*(x1**2 - x0**2)/2

            elif self.interpolation[k] == 3:
                # Linear-log
                logx = np.log(x1/x0)
                m = (y1 - y0)/logx
                partial_sum[i_low:i_high] = y0 + m*(x1*(logx - 1) + x0)

            elif self.interpolation[k] == 4:
                # Log-linear
                m = np.log(y1/y0)/(x1 - x0)
                partial_sum[i_low:i_high] = y0/m*(np.exp(m*(x1 - x0)) - 1)

            elif self.interpolation[k] == 5:
                # Log-log
                m = np.log(y1/y0)/np.log(x1/x0)
                partial_sum[i_low:i_high] = y0/((m + 1)*x0**m)*(
                    x1**(m + 1) - x0**(m + 1))

            i_low = i_high

        return np.concatenate(([0.], np.cumsum(partial_sum)))

    @classmethod
    def from_ace(cls, ace, idx=0, convert_units=True):
        """Create a Tabulated1D object from an ACE table.

        Parameters
        ----------
        ace : openmc.data.ace.Table
            An ACE table
        idx : int
            Offset to read from in XSS array (default of zero)
        convert_units : bool
            If the abscissa represents energy, indicate whether to convert MeV
            to eV.

        Returns
        -------
        openmc.data.Tabulated1D
            Tabulated data object

        """

        # Get number of regions and pairs
        n_regions = int(ace.xss[idx])
        n_pairs = int(ace.xss[idx + 1 + 2*n_regions])

        # Get interpolation information
        idx += 1
        if n_regions > 0:
            breakpoints = ace.xss[idx:idx + n_regions].astype(int)
            interpolation = ace.xss[idx + n_regions:idx + 2*n_regions].astype(int)
        else:
            # 0 regions implies linear-linear interpolation by default
            breakpoints = np.array([n_pairs])
            interpolation = np.array([2])

        # Get (x,y) pairs
        idx += 2*n_regions + 1
        x = ace.xss[idx:idx + n_pairs].copy()
        y = ace.xss[idx + n_pairs:idx + 2*n_pairs].copy()

        if convert_units:
            x *= EV_PER_MEV

        return Tabulated1D(x, y, breakpoints, interpolation)


class Tabulated2D:
    """Metadata for a two-dimensional function.

    This is a dummy class that is not really used other than to store the
    interpolation information for a two-dimensional function. Once we refactor
    to adopt GNDS-like data containers, this will probably be removed or
    extended.

    Parameters
    ----------
    breakpoints : Iterable of int
        Breakpoints for interpolation regions
    interpolation : Iterable of int
        Interpolation scheme identification number, e.g., 3 means y is linear in
        ln(x).

    """
    def __init__(self, breakpoints, interpolation):
        self.breakpoints = breakpoints
        self.interpolation = interpolation


class Sum:
    """Sum of other callables.

    Used for redundant reactions, whose cross section is defined as the sum of
    other cross sections.

    Parameters
    ----------
    functions : Iterable of Callable
        Functions which are to be added together

    Attributes
    ----------
    functions : list of Callable
        Functions which are to be added together

    """

    def __init__(self, functions):
        self.functions = list(functions)

    def __call__(self, x):
        return sum(f(x) for f in self.functions)
