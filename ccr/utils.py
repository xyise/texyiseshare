from datetime import datetime
from typing import List, TypeVar
import numpy as np
import pandas as pd

T = TypeVar("T", np.ndarray, float)


def df_to_yield(df: T, ttm: float) -> T:
    return -np.log(df) / ttm


def generate_schedule(start_dt: datetime, end_dt: datetime, freq: str) -> List[datetime]:

    freq_unit = freq[-1]
    freq_val = int(freq[:-1])
    if freq_unit == "D":
        offset_unit = "days"
    elif freq_unit == "M":
        offset_unit = "months"
    elif freq_unit == "Y":
        offset_unit = "years"
    else:
        raise ValueError(f"Unknown frequency unit {freq_unit}")

    date_offset = pd.DateOffset(**{offset_unit: freq_val})

    sched = [start_dt]
    while sched[-1] < end_dt:
        sched.append((pd.Timestamp(sched[-1]) + date_offset).to_pydatetime())

    return sched
