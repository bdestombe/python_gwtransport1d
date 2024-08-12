import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gwtransport1d.residence_time import residence_time_retarded


def compute_deposition(
    cout, flow, aquifer_pore_volume, porosity, thickness, retardation_factor, nullspace_objective="squared_lengths"
):
    """
    Compute the deposition given the added concentration of the compound in the extracted water.

    Parameters
    ----------
    cout : pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    porosity : float
        Porosity of the aquifer [dimensionless].
    thickness : float
        Thickness of the aquifer [m].
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].
    nullspace_objective : str or callable, optional
        Objective to minimize in the nullspace. If a string, it should be either "squared_lengths" or "summed_lengths". If a callable, it should take the form `objective(x, xLS, colsOfNullspace)`. Default is "squared_lengths".

    Returns
    -------
    pandas.Series
        Deposition of the compound in the aquifer [ng/m2/day].
    """
    # concentration extracted water is coeff dot deposition
    _, coeff = deposition_coefficients(
        flow, aquifer_pore_volume, porosity=porosity, thickness=thickness, retardation_factor=retardation_factor
    )

    # Underdetermined least squares solution
    deposition_ls, *_ = np.linalg.lstsq(coeff, cout, rcond=None)

    # Nullspace -> multiple solutions exist, deposition_ls is just one of them
    colsOfNullspace = nullspace(coeff)
    nullrank = colsOfNullspace.shape[1]

    # Pick a solution in the nullspace that meets new objective
    def objective(x, xLS, colsOfNullspace):
        sols = xLS + colsOfNullspace @ x
        return np.square(sols[1:] - sols[:-1]).sum()

    deposition_0 = np.zeros(nullrank)
    res = minimize(objective, x0=deposition_0, args=(deposition_ls, colsOfNullspace), method="BFGS")

    if not res.success:
        msg = f"Optimization failed: {res.message}"
        raise ValueError(msg)

    # Squared lengths is stable to solve, thus good starting point
    if nullspace_objective != "squared_lengths":
        if nullspace_objective == "summed_lengths":

            def objective(x, xLS, colsOfNullspace):
                sols = xLS + colsOfNullspace @ x
                return np.abs(sols[1:] - sols[:-1]).sum()

            res = minimize(objective, x0=res.x, args=(deposition_ls, colsOfNullspace), method="BFGS")
        elif callable(nullspace_objective):
            res = minimize(nullspace_objective, x0=res.x, args=(deposition_ls, colsOfNullspace), method="BFGS")
        else:
            msg = f"Unknown nullspace objective: {nullspace_objective}"
            raise ValueError(msg)

    deposition_data = deposition_ls + colsOfNullspace @ res.x
    return pd.Series(data=deposition_data, index=flow.index, name="deposition")


def compute_concentration(deposition, flow, aquifer_pore_volume, porosity, thickness, retardation_factor):
    """
    Compute the concentration of the compound in the extracted water given the deposition.

    Parameters
    ----------
    deposition : pandas.Series
        Deposition of the compound in the aquifer [ng/m2/day].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    porosity : float
        Porosity of the aquifer [dimensionless].
    thickness : float
        Thickness of the aquiifer [m].

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    """
    _, coeff = deposition_coefficients(
        flow, aquifer_pore_volume, porosity=porosity, thickness=thickness, retardation_factor=retardation_factor
    )
    rt = residence_time_retarded(flow, aquifer_pore_volume, retardation_factor=retardation_factor)
    valid_rt_mask = rt.notnull()
    return pd.Series(coeff @ deposition, index=flow.index[valid_rt_mask], name="cout")


def deposition_coefficients(flow, aquifer_pore_volume, porosity, thickness, retardation_factor):
    """
    Compute the coefficients of the deposition model.

    Parameters
    ----------
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    porosity : float
        Porosity of the aquifer [dimensionless].
    thickness : float
        Thickness of the aquifer [m].
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the residence time of the retarded compound in the aquifer [days].
    numpy.ndarray
        Coefficients of the deposition model [m2/day].
    """
    rt = residence_time_retarded(flow, aquifer_pore_volume, retardation_factor=retardation_factor)
    valid_rt_mask = rt.notnull()
    df_out = pd.DataFrame(data={"rt": rt[valid_rt_mask].copy(), "flow": flow[valid_rt_mask].copy()})
    df_out["dates_infiltration_retarded"] = df_out.index - pd.to_timedelta(df_out.rt, unit="D")

    # Aquifer area cathing deposition
    df_out["darea"] = df_out.flow / (retardation_factor * porosity * thickness)

    # Compute coefficients
    nin = len(flow)
    nout = len(df_out)
    dt = np.zeros((nout, nin), dtype=float)

    for iout, (date_extraction, row) in enumerate(df_out.iterrows()):
        itinf = flow.index.searchsorted(row.dates_infiltration_retarded)  # partial day
        itextr = flow.index.searchsorted(date_extraction)  # whole day
        dt[iout, itinf : itextr + 1] = 1.0

        # fraction of first day
        dt[iout, itinf] = -(row.dates_infiltration_retarded - flow.index[itinf]) / pd.to_timedelta(1.0, unit="D")

    coeff = (df_out.darea / df_out.flow).values[:, None] * dt
    return df_out, coeff


def nullspace(coefficients, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    coefficients : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    numpy.ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.

    Notes
    -----
    Source: https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
    """
    coefficients = np.atleast_2d(coefficients)
    _, s, vh = np.linalg.svd(coefficients)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    return vh[nnz:].conj().T
