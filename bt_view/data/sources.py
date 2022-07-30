'''Contains objects to connect with financial data sources'''
import pandas as pd
import datetime as dt
import pandas_datareader as pdr
import numpy as np
import quandl
import streamlit as st
from typing import List, Union
import requests, json
import types
from enum import Enum

#Move this to .env (eventually)
QUOTEMEDIA_API_KEY = 'arnsoxD1joP6rGxV_Uxy'
ALPHAVANTAGE_API_KEY = '0F8NGY6EKQZ7OOVT'

class Field(Enum):
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    DIVIDEND = 'dividend'
    SPLIT = 'split'
    ADJUSTED_OPEN = 'adj_open'
    ADJUSTED_HIGH = 'adj_high'
    ADJUSTED_LOW = 'adj_low'
    ADJUSTED_CLOSE = 'adj_close'
    ADJUSTED_VOLUME = 'adj_volumne'

class ReturnType(Enum):
    CLOSE_TO_CLOSE = 'c2c'
    OPEN_TO_OPEN = 'o2o'
    OPEN_TO_CLOSE = 'o2c'
    CLOSE_TO_OPEN = 'c2o'
    DURING_TRADING_HOURS = 'o2c'
    OUTSIDE_TRADING_HOURS = 'c2o'

def concat_fama_french_data(ff_data: pd.DataFrame, time_series: pd.DataFrame):
    """
    Helper function to concatenate price data and Fama French data.

    Args:
        ff_data: Data returned from :func: FamaFrench.get
        time_series: table with a datetime index

    Returns:
        Concatenated data
    
    """
    if isinstance(ff_data.index, pd.PeriodIndex):
        ff_data.index = ff_data.index.to_timestamp()

    data = pd.concat([time_series, ff_data], axis=1)
    tickers = time_series.columns
    tdata = data.fillna(method='ffill').dropna().drop_duplicates(subset=data.columns.drop(tickers))
    tdata.loc[:, tickers] = np.log(tdata[tickers]/tdata[tickers].shift())*100
    final = tdata.dropna()
    return final

def quotemedia_metadata(api_key) -> dict:
    """Pulls QUOTEMEDIA metadata e.g. table columns"""
    r = requests.get('https://data.nasdaq.com/api/v3/datatables/QUOTEMEDIA/PRICES/metadata?api_key={}'.format(api_key), stream=True)
    data = json.loads(r.content)['datatable']
    return data

def to_price_series(data: pd.DataFrame, field: Union[str, Field] = 'adj_close', ref_index: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    Takes the raw data output from the QuoteMedia API and transforms it
    to a time series format.

    Args:
        data: Output from :func: Eod.get.
        field: The price field to use. Default is adjusted close prices.
            See Eod.fields for a list of all fields.
        ref_index: Index to align data with. Some data, such as economic data,
            is released on days where the market is closed.

    Returns:
        Price data in time series format.
    """

    if isinstance(field, Field):
        field = field.value

    prices = data.pivot(index='date', columns='ticker')[field]

    if ref_index is not None:
        daily = prices.asfreq('d').ffill()
        idx = daily.index.intersection(ref_index)
        return daily.loc[idx]
    
    return prices

def to_returns_series(data: pd.DataFrame, return_type: Union[str, ReturnType] = 'c2c', nperiods: int = 1, lag: int = 0) -> pd.DataFrame:
    """
    Convert prices to log returns. Adjusted prices are used to calculate returns.
    
    Args:
        data: prices returned from quandl api call
        return type: how to compute returns. 
        nperiods: number of periods to compute returns over e.g. a value of 1 yields daily returns
        lag: number of periods to lag returns e.g. a value of 1 attributes today's returns to yesterday

    Returns:
        Time series of stock data returns.

    """

    if isinstance(return_type, ReturnType):
        return_type = return_type.value

    nperiods = max(nperiods, 1)
    srs_list = []
    tickers = data['ticker'].unique()

    for t in tickers:
        subset = data[data['ticker']==t].set_index('date').sort_index()

        if return_type=='c2c':
            prc = subset['adj_close']
            rets = np.log(prc/prc.shift(nperiods))
        
        elif return_type=='o2o':
            prc = subset['adj_open']
            rets = np.log(prc/prc.shift(nperiods))

        elif return_type=='o2c':
            rets = np.log(
                subset['adj_close']/subset['adj_open'].shift(nperiods-1)
            )
        
        elif return_type=='c2o':
            rets = np.log(
                subset['adj_open']/subset['adj_close'].shift(nperiods)
            )
            
        rets.name = t
        srs_list.append(rets)
        df = pd.concat(srs_list, axis=1)
        df = df.shift(-1*lag)
    
    return df


class Eod:

    """
    Pulls Daily OHLCV Data for US Equities from QuoteMedia.
    """

    def __init__(self, api_key=QUOTEMEDIA_API_KEY) -> None:
        quandl.ApiConfig.api_key = api_key
        meta_data = quotemedia_metadata(api_key)
        cols = pd.DataFrame(meta_data['columns'])
        
        self._fields = cols.loc[cols['type']=='double', 'name']
        self._dflt_end_date = dt.datetime.now()
        self._dflt_start_date = self._dflt_end_date - dt.timedelta(days=180)
        self._meta = None

    @property
    def meta_data(self) -> pd.DataFrame:
        """Returns info on all available tickers."""
        if self._meta is None:
            data = quandl.get_table('QUOTEMEDIA/TICKERS', paginate=True)
            data = data[~data['company_name'].str.contains('TEST')]
            data['Display'] = data['ticker'] + ': ' + data['company_name'] + ' (' + data['exchange'] + ')'
            self._meta = data
        return self._meta

    @property
    def fields(self) -> List[str]:
        """Returns list of fields e.g. OHLC, volume, splits, dividends."""
        return self._fields.to_list()

    def get(self, tickers: Union[str, List[str]], start: dt.datetime = None, end: dt.datetime = None) -> pd.DataFrame:
        """
        Gets data from QuoteMedia API. The resulting data may need to be transformed to be useful.
        See :func: Eod.to_price_series and :func: Eod.to_returns_series.

        Args:
            tickers: Ticker(s) of stocks and ETFs e.g. SPY, QQQ
            start: Starting date of the dataset.
            end: Final date of the dataset.

        Returns:
            Price data.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        if any([x is None for x in [start, end]]):
            data = quandl.get_table('QUOTEMEDIA/PRICES', ticker=tickers, paginate=True)
        else:
            dates = [
                x.isoformat() for x in pd.date_range(start=start, end=end, freq='d')
            ]
            data = quandl.get_table('QUOTEMEDIA/PRICES', ticker=tickers, date=dates, paginate=True)
            
        data.to_price_series = types.MethodType(to_price_series, data)
        data.to_returns_series = types.MethodType(to_returns_series, data)
        
        return data

    def tickers_select_element(self, verbose_tickers=False, key=None, label=None):

        if verbose_tickers:
            disp = 'Display'
        else:
            disp = 'ticker'

        label = label or 'Tickers'
        fmt_func = lambda idx: self.meta_data.loc[idx, disp]
        idx = st.multiselect(label, self.meta_data.index, format_func=fmt_func, key=key)
        tickers = self.meta_data.loc[idx, 'ticker']

        return tickers

    def prices_interface(self, verbose_tickers=False, key=None) -> pd.DataFrame:
        c1, c2 = st.columns((3,1,1))

        with c1:
            tickers = self.tickers_select_element(verbose_tickers, key='ticker_select_{}'.format(key))

        with c2:
            max_value = dt.datetime.today()
            min_value = dt.datetime(1995,1,1)

            start_date = st.date_input('Start Date', min_value=min_value, max_value=max_value, key='eod_start_date_{}'.format(key))
            end_date = st.date_input('End Date', min_value=min_value, max_value=max_value, key='eod_end_date_{}'.format(key))

        return tickers, start_date, end_date

    def prices_interface_condensed(self, verbose_tickers=False, key=None, label=None):
        tickers = self.tickers_select_element(verbose_tickers, key, label)
        return tickers


class FamaFrench:

    def __init__(self):
        self._dataset_names = pdr.famafrench.FamaFrenchReader('').get_available_datasets()
        self._default_start = dt.datetime(2000,1,1)

    @property
    def meta_data(self):
        return self._dataset_names
        
    def get(self, x: str, **kwargs) -> dict:
        if x in self.meta_data:
            data = pdr.get_data_famafrench(x, **kwargs)
            data.concat_to_time_series = types.MethodType(concat_fama_french_data, data)
            return data
        else:
            raise KeyError('{} is not an valid Fama French dataset'.format({x}))




