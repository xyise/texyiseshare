from datetime import datetime
from typing import Dict, List
from dnr.ccr.market import Market
import numpy as np


class Pricer:

    @property
    def id(self) -> str:
        raise NotImplementedError

    def get_pv(self, market: Market) -> np.ndarray:
        raise NotImplementedError

    def get_fixing_requirements(self) -> Dict[str, List[datetime]]:
        raise NotImplementedError
    
    def update(self, market: Market) -> None:
        return

    @property
    def valuation_currency(self) -> str:
        raise NotImplementedError
