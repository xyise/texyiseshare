from datetime import datetime, timedelta
from typing import Dict, List
from dnr.ccr.constants import DAYS_IN_ANNUM
from dnr.ccr.market import Market
from dnr.ccr.pricers.pricer import Pricer
import numpy as np
import pandas as pd


class YieldRate(Pricer):

    def __init__(self, id: str, currency: str, tenor: str):

        self._id = id
        self.ccy = currency
        self.tenor = tenor

    @property
    def id(self) -> str:
        return self._id

    def get_pv(self, market: Market) -> float:
        
        mat_dt = (
                    pd.to_datetime(market.as_of_dt) + pd.DateOffset(months=int(self.tenor[:-1]))
                ).to_pydatetime()
        yr_frac = (mat_dt - market.as_of_dt) / timedelta(days=DAYS_IN_ANNUM)
        disc_fac = market.get_discount_factor(self.ccy, mat_dt)
        
        return - np.log(disc_fac) / yr_frac

    def get_fixing_requirements(self) -> Dict[str, List[datetime]]:
        return {}

    @property
    def valuation_currency(self) -> str:
        return self.ccy