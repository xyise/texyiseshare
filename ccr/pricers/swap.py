from datetime import datetime
from typing import Dict, List
from dnr.ccr.constants import DAYS_IN_ANNUM
from dnr.ccr.pricers.pricer import Pricer
from dnr.ccr.utils import generate_schedule
import numpy as np
import pandas as pd

from dnr.ccr.market import Market


class Swap(Pricer):
    def __init__(
        self, id: str, currency: str, start_dt: datetime, end_dt: datetime, fixed_freq: str, float_freq: str, rate: float, notional: float
    ):
        "notional: positive means payer, negative means receiver"

        self._id = id
        self.ccy = currency
        self.start_dt = start_dt
        self.end_dt = end_dt

        self.fixed_freq = fixed_freq
        self.float_freq = float_freq

        self.rate = rate

        self.notional = notional

        fixed_sched = generate_schedule(start_dt, end_dt, fixed_freq)
        float_sched = generate_schedule(start_dt, end_dt, float_freq)

        self.fixed_pay_sched = fixed_sched[1:]
        self.fixed_year_fracs = [
            (fixed_sched[i + 1] - fixed_sched[i]).days / DAYS_IN_ANNUM for i in range(len(fixed_sched) - 1)
        ]

        self.float_pay_sched = float_sched[1:]
        self.float_ac_start_sched = float_sched[:-1]
        self.float_ac_end_sched = float_sched[1:]

        self.float_year_fracs = [
            (float_sched[i + 1] - float_sched[i]).days / DAYS_IN_ANNUM for i in range(len(float_sched) - 1)
        ]

        # fixing dates
        self.float_fixing_sched = float_sched[:-1]
        self.fixing_name = f"RATE_{currency}_{float_freq}"

        # caching for performance
        self._fixed_sched_ts = np.array([dt.timestamp() for dt in self.fixed_pay_sched])
        self._float_sched_ts = np.array([dt.timestamp() for dt in self.float_pay_sched])

    @property
    def id(self) -> str:
        return self._id

    def get_pv(self, market: Market) -> np.ndarray:

        return self.get_fixed_pv(market) - self.get_float_pv(market)

    def get_fixed_pv(self, market: Market) -> np.ndarray:

        as_of_ts = market.as_of_dt.timestamp()

        fixed_start_idx = np.digitize(as_of_ts, self._fixed_sched_ts)
        fixed_pv = 0.0
        for i in range(fixed_start_idx, len(self.fixed_pay_sched)):
            fixed_pv += self.fixed_year_fracs[i] * market.get_discount_factor(self.ccy, self.fixed_pay_sched[i])
        fixed_pv *= self.rate

        return self.notional * fixed_pv
    
    def get_float_pv(self, market: Market) -> np.ndarray:

        as_of_ts = market.as_of_dt.timestamp()

        float_start_idx = np.digitize(as_of_ts, self._float_sched_ts)
        float_pv = 0.0
        for i in range(float_start_idx, len(self.float_pay_sched)):
            fixing_dt = self.float_fixing_sched[i]
            if fixing_dt <= market.as_of_dt:
                fixing_rate = market.get_fixing(self.float_fixing_sched[i], self.fixing_name)
            else:
                fixing_rate = (
                    market.get_discount_factor(self.ccy, self.float_ac_start_sched[i])
                    / market.get_discount_factor(self.ccy, self.float_ac_end_sched[i])
                    - 1.0
                ) / self.float_year_fracs[i]

            float_pv += fixing_rate * self.float_year_fracs[i] * market.get_discount_factor(self.ccy, self.float_pay_sched[i])

        return self.notional * float_pv

    def get_fixing_requirements(self) -> Dict[str, List[datetime]]:
        return {self.fixing_name: self.float_fixing_sched}

    @property
    def valuation_currency(self) -> str:
        return self.ccy