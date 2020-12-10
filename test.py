import pickle
import numpy.testing
from pandas import testing
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype

class TestDataframe():

    @classmethod
    def setup_class(cls):
        cls.col_names = pickle.load(open('test_file/col_names.pk', 'rb'))
        cls.df = pickle.load(open('question/dataframe.pk', 'rb'))

    def test_df_shape(self):
        assert self.df.shape == (3280, 19)

    def test_num_cols(self):
        assert len(self.col_names) == len(self.df.columns)

    def test_cols_names(self):
        cols = self.df.columns
        assert np.array_equiv(self.col_names, np.sort(cols))

    def test_col_type(self):
        for col in self.col_names:
            assert is_numeric_dtype(self.df[col])

    def test_index(self):
        assert is_datetime64_dtype(self.df.index)

    def test_sanity_check_avg_wind_speed(self):
        mean = round(np.mean(self.df["2011-08-1":"2011-08-20"]["Average windspeed (mph)"]), 2)
        assert mean == 4.64

    def test_sanity_check_std_min_temp(self):
        std1 = round(np.std(self.df["2011-04-20":"2012-01-1"]["Minimum temperature (°F)"]), 2)
        assert std1 == 11.79

    def test_sanity_check_std_max_pressure(self):
        std2 = round(np.std(self.df["2011-04-20":"2012-01-1"]["Maximum pressure"]), 2)
        assert std2 == 0.27

    def test_max_temp(self):
        max = round(np.max(self.df["2011-04-20":"2012-01-1"]["Maximum temperature (°F)"]), 2)
        assert max == 89.7
