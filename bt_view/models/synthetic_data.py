import pandas as pd
import numpy as np
import datetime as dt
from string import ascii_uppercase
from random import choices

def generate_data(start_date=dt.datetime(2016,1,1), end_date=dt.datetime(2022,12,31), num_securities=5):
    col_names = [''.join(choices(ascii_uppercase, k=4)) for i in range(num_securities)]
    idx = pd.date_range(start_date, end_date, freq='d')
    daily_vol = 0.005
    dim = (len(idx), num_securities)
    df = (1+pd.DataFrame(np.random.normal(0, daily_vol, dim), index=idx)).cumprod()*100
    df.columns = col_names
    return df