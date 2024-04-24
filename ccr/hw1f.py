from datetime import datetime
from typing import List
from dnr.ccr.constants import DAYS_IN_ANNUM
import numpy as np
from scipy.interpolate import interp1d


class HW1FModel:

    def __init__(self, calib_dt: datetime, mr: float, sigma: float, phi_dates: List[datetime], phi_values: List[float]):

        self.calib_dt = calib_dt
        self.mr = mr
        self.sigma = sigma
        self.phi_date_list = phi_dates
        self.phi_value_list = phi_values

        # Interpolation function on phi and its integral
        phi_taus = np.array([self.time_from_calib_dt(dt) for dt in phi_dates])
        phi_vals = np.array(phi_values)
        self.phi_func = interp1d(
            phi_taus, phi_vals, kind="next", fill_value=(phi_vals[0], phi_vals[-1]), bounds_error=False
        )

        # Integral of phi
        phi_taus_ext = np.hstack((0.0, phi_taus))
        I_phi = np.cumsum(np.diff(phi_taus_ext) * phi_vals)
        I_phi_ext = np.hstack((0.0, I_phi))
        self.I_phi_func = interp1d(phi_taus_ext, I_phi_ext, kind="linear", fill_value="extrapolate")

    def time_from_calib_dt(self, dt: datetime) -> float:
        return (dt - self.calib_dt).days / DAYS_IN_ANNUM

    def get_discount_factor(self, t: float, T: float, x_t: np.ndarray) -> np.ndarray:
        "vectorised calculations"

        a, s = self.mr, self.sigma
        t2T = T - t

        I_phi_tT = self.I_phi_func(T) - self.I_phi_func(t)
        U_tT = (s / a) ** 2 * (t2T - 2.0 / a * (1 - np.exp(-a * t2T)) + 1 / (2 * a) * (1 - np.exp(-2 * a * t2T)))
        B_tT = (1 - np.exp(-a * t2T)) / a
        A_tT = -I_phi_tT + 0.5 * U_tT

        return np.exp(A_tT - B_tT * x_t)

    def get_r(self, t: float, x_t: np.ndarray) -> np.ndarray:

        return self.phi_func(t) + x_t

    def simulate(self, time_step: float, x_t: np.ndarray, dZ: np.ndarray) -> np.ndarray:

        return x_t - self.mr * time_step * x_t + self.sigma * np.sqrt(time_step) * dZ
