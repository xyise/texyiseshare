from asyncio import as_completed
from datetime import datetime
from typing import Dict, List
from dnr.ccr.market import Market
from dnr.ccr.pricers import swap
from dnr.ccr.pricers.pricer import Pricer
from dnr.ccr.pricers.swap import Swap
import numpy as np
from scipy.stats import norm

class EuropeanSwaption(Pricer):

    def __init__(
            self, id: str, currency: str, is_payer: bool,
            expiry_dt: datetime, strike: float,
            swap_end_dt: datetime, notional: float):
        "notional: positive means payer, negative means receiver"
        

        self._id = id
        self.ccy = currency
        self.is_payer = is_payer
        
        self.expiry_dt = expiry_dt
        self.strike = strike
        self.swap_end_dt = swap_end_dt
        self.notional = notional

        self.underlying_swap = Swap(
            f"{id}_SWAP", currency, expiry_dt, swap_end_dt, "6M", "3M", strike, 1
        )

        self.exercised: np.ndarray = None
    @property
    def id(self) -> str:
        return self._id
    

    def get_pv(self, market: Market) -> np.ndarray:

        if market.as_of_dt >= self.expiry_dt:
            swap_pv = self.underlying_swap.get_pv(market) * (-1 if self.is_payer else 1) * self.notional
            return swap_pv * self.exercised
        
        annuity = self.underlying_swap.get_fixed_pv(market) / self.strike
        swap_rate = self.underlying_swap.get_float_pv(market) / annuity
        swaption_vol = 0.05

        DF = market.get_discount_factor(self.ccy, self.expiry_dt)
        B = self._black_undisc(swap_rate, self.strike, swaption_vol, (self.swap_end_dt - self.expiry_dt).days / 365, self.is_payer)
        return annuity * DF * B * self.notional
    
    def _black_undisc(self, fwd, strike, vol, tte, is_call):
        d1 = (np.log(fwd / strike) + 0.5 * vol ** 2 * tte) / (vol * np.sqrt(tte))
        d2 = d1 - vol * np.sqrt(tte)
        if is_call:
            return fwd * norm.cdf(d1) - strike * norm.cdf(d2)
        else:
            return strike * norm.cdf(-d2) - fwd * norm.cdf(-d1)

    
    def get_fixing_requirements(self) -> Dict[str, List[datetime]]:
        return self.underlying_swap.get_fixing_requirements()
    
    def update(self, market: Market) -> None:

        if market.as_of_dt != self.expiry_dt:
            return
        
        swap_pv = self.underlying_swap.get_pv(market)
        if self.is_payer:
            self.exercised = (swap_pv <= 0) * 1.0
        else:
            self.exercised = (swap_pv > 0) * 1.0

    @property
    def valuation_currency(self) -> str:
        return self.ccy