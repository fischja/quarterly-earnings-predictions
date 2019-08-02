import numpy as np


class YearQuarter:
    """An immutable wrapper to simplify operations on (year, quarter) date indices."""

    def __init__(self, year, quarter):
        accepted_quarters = [1, 2, 3, 4]
        if quarter not in accepted_quarters:
            raise ValueError(f'quarter must be an element of {accepted_quarters}')
        if year < 0:
            raise ValueError(f'year cannot be negative')
        self._year = year
        self._quarter = quarter

    @property
    def year(self):
        return self._year

    @property
    def quarter(self):
        return self._quarter

    def get_next(self, n_quarters=1):
        quarter = self._quarter + (n_quarters % 4)
        if quarter > 4:
            quarter = quarter - 4

        year = self._year + np.floor(n_quarters / 4).astype(int)
        if quarter < self._quarter:
            year += 1

        return YearQuarter(year, quarter)

    def get_prev(self, n_quarters=1):
        quarter = self._quarter - (n_quarters % 4)
        if quarter <= 0:
            quarter = quarter + 4

        year = self._year - np.floor(n_quarters / 4).astype(int)
        if quarter > self._quarter:
            year -= 1

        return YearQuarter(year, quarter)

    def equals(self, other):
        if not isinstance(other, YearQuarter):
            return False

        return self._quarter == other.quarter and self._year == other.year

    def abs_quarters_diff(self, other):
        if not isinstance(other, YearQuarter):
            raise ValueError('other is not an instance of YearQuarter')

        if self._year == other.year:
            return abs(self._quarter - other.quarter)

        num_quarters = max(0, 4 * (abs(self._year - other.year) - 1))
        if self._year > other.year:
            num_quarters += self._quarter
            num_quarters += (4 - other.quarter)
        else:
            num_quarters += other.quarter
            num_quarters += (4 - self._quarter)

        return num_quarters

    def clone(self):
        return YearQuarter(self._year, self._quarter)
