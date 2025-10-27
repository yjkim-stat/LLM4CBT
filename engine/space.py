import logging
from typing import List, Dict


logger = logging.getLogger(__name__)


class Space:
    def __init__(self, scope: List[str]) -> None:
        self.variables = {k:'' for k in scope}

    def __getitem__(self, idx):
        return self.variables[idx]
    
    def __setitem__(self, key, value):
        self.variables[key] = value

    def sync(self, new_vals: Dict):
        for var_name in self.names:
            if var_name in new_vals.keys():
                logger.debug(f'Update {var_name} : {self[var_name]} --> {new_vals[var_name]}')
                self[var_name] = new_vals[var_name]

    @property
    def names(self):
        return list(self.variables.keys())

    @property
    def values(self):
        return self.variables    

    def __str__(self) -> str:
        res = ""
        for k, v in self.variables.items():
            if v is not None:
                res += f'{k}:{v}\n'
        return res

class DiagnosisSpace(Space):
    def __init__(self, keys) -> None:
        super().__init__(keys)
        for key in keys:
            self.variables[key] = ""