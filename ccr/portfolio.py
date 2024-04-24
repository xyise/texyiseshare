from datetime import date, datetime
from typing import Dict, List, Set

from dnr.ccr.pricers.pricer import Pricer
import pandas as pd


class Portfolio:

    def __init__(self, pricers: List[Pricer]):

        self.pricers = pricers

        fixing_requirement_map: Dict[datetime, Set[str]] = {}
        for pricer in pricers:
            for fixing_name, fixing_dates in pricer.get_fixing_requirements().items():
                for fixing_date in fixing_dates:
                    if fixing_date not in fixing_requirement_map:
                        fixing_requirement_map[fixing_date] = set()
                    fixing_requirement_map[fixing_date].add(fixing_name)

        self.fixing_requirement_map = fixing_requirement_map

    def get_historical_fixing_requirement(self, as_of_dt: datetime) -> Dict[str, List[datetime]]:
        hist_fixing_req_map: Dict[str, List[datetime]] = {}
        for fixing_date, fixing_names in self.fixing_requirement_map.items():
            if fixing_date <= as_of_dt:
                for fixing_name in fixing_names:
                    if fixing_name not in hist_fixing_req_map:
                        hist_fixing_req_map[fixing_name] = []
                    hist_fixing_req_map[fixing_name].append(fixing_date)
        return hist_fixing_req_map

    def get_fixing_requirement(self, as_of_dt: datetime) -> Set[str]:
        return self.fixing_requirement_map.get(as_of_dt, set())

    def get_trade_ids(self) -> List[str]:
        trade_ids = []
        for pricer in self.pricers:
            trade_ids.append(pricer.id)
        return trade_ids