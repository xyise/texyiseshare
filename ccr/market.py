from datetime import datetime, timedelta
from typing import Dict, List, Set, Union

from dnr.ccr.constants import DAYS_IN_ANNUM
from dnr.ccr.fx_gbm import FXGBMModel
from dnr.ccr.hw1f import HW1FModel
from dnr.ccr.utils import df_to_yield
import numpy as np
import pandas as pd


class Market:

    @property
    def calib_dt(self) -> datetime:
        pass

    @property
    def as_of_dt(self) -> datetime:
        pass

    def get_discount_factor(self, ccy: str, mat_dt: datetime) -> np.ndarray:
        pass

    def get_fx_rate(self, ccy1: str, ccy2: str) -> np.ndarray:
        pass

    def set_historical_fixings(self, historical_fixing_req_map: Dict[str, List[datetime]]) -> None:
        pass

    def set_modelled_fixings(self, index_names: Set[str]) -> None:
        pass

    def get_fixing(self, fixing_date: datetime, index_name: str) -> np.ndarray:
        pass

    def evolve(self, next_dt: datetime) -> None:
        pass

    def get_state_df(self) -> pd.DataFrame:
        pass


class MarketXCCYHW1F(Market):

    def __init__(
        self,
        ccy1: str,
        hw1f1: HW1FModel,
        ccy2: str,
        hw1f2: HW1FModel,
        fx_bgm_12: FXGBMModel,
        num_scens: int,
        rng: np.random.Generator,
    ):
        self.ccy1 = ccy1
        self.ccy2 = ccy2
        self.rate_model_map = {ccy1: hw1f1, ccy2: hw1f2}
        self.fx_model = fx_bgm_12
        self.nscens = num_scens
        self.rng = rng

        self._as_of_dt = hw1f1.calib_dt

        self._as_of_state_map = {
            ccy1: np.zeros(num_scens),
            ccy2: np.zeros(num_scens),
            ccy1 + ccy2: fx_bgm_12.x0 * np.ones(num_scens),
        }
        self.fixing_map: Dict[str, Dict[datetime, float | np.ndarray]] = {}

    @property
    def as_of_dt(self) -> datetime:
        return self._as_of_dt

    @property
    def calib_dt(self) -> datetime:
        return self.rate_model_map[self.ccy1].calib_dt

    def set_historical_fixings(self, historical_fixing_req_map: Dict[str, List[datetime]]) -> None:
        for fixing_name, fixing_dates in historical_fixing_req_map.items():
            if fixing_name not in self.fixing_map:
                self.fixing_map[fixing_name] = {}

            # hard code for now
            if fixing_name.startswith("FX"):
                for fixing_date in fixing_dates:
                    if fixing_name.split("_")[1] == self.ccy1 + self.ccy2:
                        self.fixing_map[fixing_name][fixing_date] = self.fx_model.x0
                    elif fixing_name.split("_")[1] == self.ccy2 + self.ccy1:
                        self.fixing_map[fixing_name][fixing_date] = 1.0 / self.fx_model.x0
                    else:
                        raise ValueError(f"Unknown FX fixing {fixing_name}")

            elif fixing_name.startswith("RATE"):
                ccy = fixing_name.split("_")[1]
                if ccy not in self.rate_model_map:
                    raise ValueError(f"Unknown currency {ccy}")
                for fixing_date in fixing_dates:
                    self.fixing_map[fixing_name][fixing_date] = self.rate_model_map[ccy].phi_func(0.0)

            else:
                raise ValueError(f"Unknown fixing {fixing_name}")

    def set_modelled_fixings(self, index_names: Set[str]) -> None:

        for index_name in index_names:
            if index_name not in self.fixing_map:
                self.fixing_map[index_name] = {}

            if self.as_of_dt in self.fixing_map[index_name]:
                continue

            if index_name.startswith("FX"):
                if index_name.split("_")[1] == self.ccy1 + self.ccy2:
                    self.fixing_map[index_name][self.as_of_dt] = self._as_of_state_map[self.ccy1 + self.ccy2]
                elif index_name.split("_")[1] == self.ccy2 + self.ccy1:
                    self.fixing_map[index_name][self.as_of_dt] = 1.0 / self._as_of_state_map[self.ccy1 + self.ccy2]
                else:
                    raise ValueError(f"Unknown FX fixing {index_name}")
            elif index_name.startswith("RATE"):
                ccy = index_name.split("_")[1]
                tenor = index_name.split("_")[2]
                if ccy not in self.rate_model_map:
                    raise ValueError(f"Unknown currency {ccy}")

                fixing_period_end_dt = (
                    pd.to_datetime(self.as_of_dt) + pd.DateOffset(months=int(tenor[:-1]))
                ).to_pydatetime()
                fixing_yr = (fixing_period_end_dt - self.as_of_dt) / timedelta(days=DAYS_IN_ANNUM)
                fixing_rate = (
                    1.0
                    / self.get_discount_factor(ccy, fixing_period_end_dt)
                    - 1.0
                ) / fixing_yr
                self.fixing_map[index_name][self.as_of_dt] = fixing_rate

            else:
                raise ValueError(f"Unknown fixing {index_name}")

    def get_fixing(self, fixing_date: datetime, index_name: str) -> np.ndarray:

        if index_name not in self.fixing_map:
            raise ValueError(f"Unknown fixing {index_name}. Set the fixing first.")

        if fixing_date not in self.fixing_map[index_name]:
            raise ValueError(f"Unknown fixing date {fixing_date} for {index_name}. Set the fixing first.")

        return self.fixing_map[index_name][fixing_date]

    def get_discount_factor(self, ccy: str, mat_dt: datetime) -> np.ndarray:
        if ccy in self.rate_model_map:
            model = self.rate_model_map[ccy]
            t = model.time_from_calib_dt(self._as_of_dt)
            T = model.time_from_calib_dt(mat_dt)
            return self.rate_model_map[ccy].get_discount_factor(t, T, self._as_of_state_map[ccy])
        else:
            raise ValueError(f"Unknown currency {ccy}")

    def get_fx_rate(self, ccy1: str, ccy2: str) -> np.ndarray:
        if ccy1 == ccy2:
            return np.ones(self.nscens)
        elif ccy1 + ccy2 == self.ccy1 + self.ccy2:
            return self._as_of_state_map[self.ccy1 + self.ccy2]
        elif ccy1 + ccy2 == self.ccy2 + self.ccy1:
            return 1.0 / self._as_of_state_map[self.ccy1 + self.ccy2]
        else:
            raise ValueError(f"Unknown FX rate {ccy1}{ccy2}")

    def evolve(self, next_dt: datetime) -> None:

        # assume zero correlation for now
        dZ_rate_map = {ccy: self.rng.normal(size=self.nscens) for ccy in self.rate_model_map.keys()}
        dZ_fx = self.rng.normal(size=self.nscens)

        for ccy in self.rate_model_map.keys():
            time_step = self.rate_model_map[ccy].time_from_calib_dt(next_dt) - self.rate_model_map[
                ccy
            ].time_from_calib_dt(self._as_of_dt)
            self._as_of_state_map[ccy] = self.rate_model_map[ccy].simulate(
                time_step, self._as_of_state_map[ccy], dZ_rate_map[ccy]
            )

        t = self.fx_model.time_from_calib_dt(self._as_of_dt)
        T = self.fx_model.time_from_calib_dt(next_dt)
        time_step = T - t
        mu = self.rate_model_map[self.ccy2].get_r(
            t, self._as_of_state_map[self.ccy2]
        ) - self.rate_model_map[self.ccy1].get_r(t, self._as_of_state_map[self.ccy1])

        ccy1ccy2 = self.ccy1 + self.ccy2
        self._as_of_state_map[ccy1ccy2] = self.fx_model.simulate(
            time_step, mu, self._as_of_state_map[ccy1ccy2], dZ_fx
        )

        self._as_of_dt = next_dt

    def get_state_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._as_of_state_map)