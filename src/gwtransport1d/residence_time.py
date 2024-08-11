import numpy as np
import pandas as pd


def residence_time_retarded(flow, aquifer_pore_volume, retardation_factor):
    """
    Compute the residence time of retarded compound in the water in the aquifer.

    Parameters
    ----------
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].

    Returns
    -------
    pandas.Series
        Residence time of the retarded compound in the aquifer [days].
    """
    flow_cum = flow.cumsum()
    dates_days_extraction = (flow.index - flow.index[0]) / np.timedelta64(1, "D")
    dates_infiltration_retarded = (
        pd.to_timedelta(
            np.interp(
                flow_cum - retardation_factor * aquifer_pore_volume,
                flow_cum,
                dates_days_extraction,
                left=np.nan,
                right=np.nan,
            ),
            unit="D",
        )
        + flow.index[0]
    )
    data = (flow.index - dates_infiltration_retarded) / np.timedelta64(1, "D")
    return pd.Series(data=data, index=flow.index, name="residence_times_retarded")
