
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict
from dnr.ccr.market import Market
from dnr.ccr.portfolio import Portfolio
import numpy as np
import pandas as pd

@dataclass
class ValuationOutput:
    risk_factor_cube_df: pd.DataFrame
    value_cube_df: pd.DataFrame

def extract_scenario_df(cube: pd.DataFrame, scenario: int) -> pd.DataFrame:
    return cube.reorder_levels([1,0], axis=0).loc[scenario]

def extract_instance_df(cube: pd.DataFrame, instance: int) -> pd.DataFrame:
    return cube[instance].unstack()

def valuate(valuation_ccy: str, portfolio: Portfolio, market: Market, nsteps: int, val_freq: int) -> ValuationOutput:
    
    calib_dt = market.calib_dt
    sim_dt_list = [calib_dt + timedelta(days=i) for i in range(nsteps)]
    val_dt_list = sim_dt_list[:-1][0::val_freq]

    risk_factor_map: Dict[datetime, pd.DataFrame] = {}
    value_map: Dict[datetime, pd.DataFrame] = {}

    hist_fixing_req = portfolio.get_historical_fixing_requirement(market.as_of_dt)
    market.set_historical_fixings(hist_fixing_req)

    for idx in range(nsteps):

        as_of_dt = sim_dt_list[idx]
        market.set_modelled_fixings(portfolio.get_fixing_requirement(as_of_dt))

        for pricer in portfolio.pricers:
            pricer.update(market)
            
        if as_of_dt in val_dt_list:        
            risk_factor_map[as_of_dt] = market.get_state_df().rename_axis("scenario", axis=0)
            
            as_of_value_map: Dict[str, np.ndarray] = {}
            for pricer in portfolio.pricers:
                as_of_value_map[pricer.id] = pricer.get_pv(market) * market.get_fx_rate(pricer.valuation_currency, valuation_ccy) 
            value_map[as_of_dt] = pd.DataFrame(as_of_value_map).rename_axis("scenario", axis=0)
            
        if idx < nsteps - 1:
            market.evolve(sim_dt_list[idx + 1])


    return ValuationOutput(
        risk_factor_cube_df=pd.concat(risk_factor_map.values(), keys=risk_factor_map.keys(), names=["as_of_dt"]),
        value_cube_df=pd.concat(value_map.values(), keys=value_map.keys(), names=["as_of_dt"])
    )
