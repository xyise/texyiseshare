from datetime import datetime

from dnr.ccr.constants import DAYS_IN_ANNUM
import numpy as np


class FXGBMModel:

    def __init__(self, calib_dt: datetime, x0: float, sigma: float):

        self.calib_dt = calib_dt
        self.x0 = x0
        self.sigma = sigma

    def simulate(self, time_step: float, mu: np.ndarray, x_t: np.ndarray, dZ: np.ndarray) -> np.ndarray:

        return x_t * np.exp((mu - 0.5 * self.sigma**2) * time_step + self.sigma * np.sqrt(time_step) * dZ)

    def time_from_calib_dt(self, dt: datetime) -> float:
        return (dt - self.calib_dt).days / DAYS_IN_ANNUM
