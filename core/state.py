import pandas as pd
import numpy as np


class State:
    def __init__(self) -> None:
        self._cols: dict[str, list[float]] = {}
        self._rows = 0

    def _ensure_col(self, name: str) -> list[float]:
        if name not in self._cols:
            # back-fill all previous rows with NaN when a new column is introduced late
            self._cols[name] = [np.nan] * (self._rows - 1)
        return self._cols[name]
    
    def next_row(self) -> None:
        """Start a new row; pre-fill all existing columns with NaN."""
        self._rows += 1
        for col in self._cols.values():
            col.append(np.nan)

    def register_value(self, name: str, value) -> None: 
        col = self._ensure_col(name)
        # Ensure this column is aligned with the current row index
        if len(col) < self._rows - 1:
            # column is behind -> fill missing previous rows with NaN
            col.extend([np.nan] * ((self._rows - 1) - len(col)))
        if len(col) == self._rows:
            # already set a value for this row -> overwrite
            col[-1] = value
        elif len(col) == self._rows - 1:
            # first value for this row -> append
            col.append(value)
        else:
            # if someone appended too far, trim to the current row then append
            col[:] = col[: self._rows - 1] + [value]

    def register_vector(self, name, vector, labels=['x', 'y', 'z']) -> None:
        for i, label in enumerate(labels):
            self.register_value(f"{name}_{label}", vector[i])

    def fill_missing(self) -> None:
        max_rows = max(len(col) for col in self._cols.values())
        for col in self._cols.values():
            while len(col) < max_rows:
                col.append(np.nan)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._cols)

    def to_csv(self, filepath: str) -> None:
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
