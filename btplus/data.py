"""
Contains objects to connect with financial data sources.

This module is meant to be a substitute for a DBMS. It contains class objects which
connect to various data sources and cache the results locally in a way that persists
beyond the runtime of the calling script. 

Once all of this data is moved to Google BigQuery, this module will be deprecated. 

All classes follow the design pattern implemented in ``Shelf``. Data is pulled from an API
using the method ``Shelf.get``. The parameter ``overwrite`` determines whether 
data is pulled by calling the API or from the cached values (if they exist).

For example, suppose we want a time series of annual GDP for the United States. We would use 
``Fred`` to retrieve this data from the FRED database.

>>> from btplus.data import Fred
>>> gdp_cached = Fred.get('GDP', overwrite=False)
>>> gdp_fresh = Fred.get('GDP', overwrite=True)
"""
import pandas as pd
import datetime as dt
import pandas_datareader as pdr
import numpy as np
import quandl
from typing import Iterable, List, Union, Dict
import requests, json
import shelve
import os
import pandas_market_calendars as mcal

from enum import Enum
from pandas_datareader._utils import RemoteDataError

#Move this to .env (eventually)
class ApiConfig:
    """
    Convenience object for managing API keys. You must either write a key here
    or pass one in when calling any ``Shelf.get`` method.
    """
    quotemedia = 'arnsoxD1joP6rGxV_Uxy'
    alphavantage = '0F8NGY6EKQZ7OOVT'
    Fred = '06c85841022970410e537ed078a3bdba'

class Shelf:

    """
    Base static class for accessing data from external APIs. Requires overriding ``Shelf.pull``
    and ``Shelf.endpoint``. The interface is similar to that of a ``dict``.
     
    """

    endpoint = 'default'
    '''str: A unique key where all of the data associated with this object will be stored'''

    @classmethod
    def _fp(cls) -> str:
        dir = os.path.dirname(__file__)
        pdir = os.path.dirname(dir)
        path = os.path.join(pdir, 'data', cls.endpoint)
        return path

    @classmethod
    def keys(cls):
        with shelve.open(cls._fp()) as db:
            return list(db.keys())

    def pull(self, symbols: Union[str, List[str]], *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def get(cls, x: Union[str, List[str]], overwrite=True, **kwargs) -> Dict[str, pd.DataFrame]:

        if isinstance(x, str):
            x = [x]
        
        symbols = set(x)
        with shelve.open(cls._fp()) as db:
            keys = set(db.keys())

            #if overwrite is False, only pull data missing from cache
            if overwrite:
                cls.pull(list(symbols), **kwargs)

            else:
                missing = list(symbols.difference(keys))
                if missing:
                    cls.pull(missing, **kwargs)
        
        with shelve.open(cls._fp()) as db: #re-opening shelve is necessary to access newly pulled data
            out = {i: db[i] for i in symbols}

        return out

    @classmethod
    def clear(cls) -> None:
        with shelve.open(cls._fp()) as db:
            db.clear()

    @classmethod
    def pop(cls, x) -> None:
        with shelve.open(cls._fp()) as db:
            db.pop(x)

class QuoteMedia(Shelf):

    """
    Access daily OHLCV data from QuoteMedia's API. 

    Examples:

        Pull data by writing a valid API key to ``ApiConfig.quotemedia`` (recommended) 
        or by passing the key as an argument to ``QuoteMedia.get``.

        >>> #pull OHLCV data for SPY in 2021
        >>> from btplus.data import ApiConfig, QuoteMedia
        >>> import datetime as dt
        >>> start = dt.datetime(2021,1,1)
        >>> end = dt.datetime(2022,1,1)
        >>> ...
        >>> #option 1: ApiConfig
        >>> ApiConfig.quotemedia = my_api_key
        >>> spy_ohlcv = QuoteMedia.get('SPY', start=start, end=end)
        >>> ...
        >>> #option 2: pass api_key as argument
        >>> spy_ohlcv = QuoteMedia.get('SPY', start=start, end=end, api_key=my_api_key)

    """

    endpoint = 'quotemedia_daily_stock'

    class ReturnType(Enum):
        CLOSE_TO_CLOSE = 'c2c'
        OPEN_TO_OPEN = 'o2o'
        OPEN_TO_CLOSE = 'o2c'
        CLOSE_TO_OPEN = 'c2o'
        DURING_TRADING_HOURS = 'o2c'
        OUTSIDE_TRADING_HOURS = 'c2o'

    def _ohlc_returns(data: pd.DataFrame, return_type: Union[str, ReturnType], same_period=False) -> pd.Series:

        if isinstance(return_type, QuoteMedia.ReturnType):
            return_type = return_type.value

        if return_type=='c2c':
            prc = data['adj_close']
            rets = np.log(prc/prc.shift())
        
        elif return_type=='o2o':
            prc = data['adj_open']
            rets = np.log(prc/prc.shift())

        elif return_type=='o2c': #if same_period, then compute returns using open and close prices from the same row
            s2 = data['adj_close']
            s1 = data['adj_open'].shift() if not same_period else data['adj_open']
            rets = np.log(
                s2/s1
            )
        
        elif return_type=='c2o':
            rets = np.log(
                data['adj_open']/data['adj_close'].shift()
            )

        return rets

    @classmethod
    def pull(cls, tickers: Union[str, List[str]], start=None, end=None, api_key=None) -> None:

        raw = get_data_quotemedia(tickers, api_key, start, end)
        tickers = raw['ticker'].unique()

        with shelve.open(cls._fp()) as db:
            for t in tickers:
                df = raw[raw['ticker']==t].drop('ticker', axis=1).sort_values('date')
                db[t] = df

    def period_returns(tickers, return_type='c2c', periods=1, **kwargs):
        """
        Convert adjusted prices to log returns over periods of arbitrary length.
        
        Args:
            tickers (str or list of str): symbols corresponding to stocks/ETFs
            return_type (str or ReturnType): How to compute returns, e.g. open-to-open, open-to-close.
            periods (int): The number of periods to compute returns over e.g. a value of 1 yields daily returns. If 
                       this is collection of dates, returns will be calculated for those intervals.
            kwargs: If False, use cached data instead of calling API.

        Returns:
            Time series of stock returns.

        Example:
            Suppose we want returns of an asset during the trading day.

            >>> from btplus.data import QuoteMedia
            >>> return_type = 'o2c' #stands for 'open-to-close'
            >>> trading_day_returns = QuoteMedia.period_returns('SPY', return_type, api_key=my_api_key)

            Now, suppose we want returns of an asset over three day periods using open prices starting in March, 2005.
            
            >>> from btplus.data import QuoteMedia
            >>> import datetime as dt
            >>> return_type = 'o2o' #stands for 'open-to-open'
            >>> start = dt.datetime(2005,3,1)
            >>> returns = QuoteMedia.period_returns('SPY', return_type, periods=3, api_key=my_api_key, start=start)

            Finally, we can also calculate returns over a completely arbitrary set of periods.

            >>> from btplus.data import QuoteMedia
            >>> import datetime as dt
            >>> return_type = 'c2c' #stands for 'close-to-close'
            >>> dates = [
                    dt.datetime(2000,4,18),
                    dt.datetime(2004, 3, 9),
                    dt.datetime(2009, 9, 17),
                    dt.datetime(2017, 5, 4)
                ]
            >>> tickers = ['SPY', 'QQQ', 'F', 'PG']
            >>> returns = QuoteMedia.period_returns(tickers, return_type, dates, api_key=my_api_key)


        """

        data = QuoteMedia.get(tickers, **kwargs)
        srs_list = []

        idx = set.intersection(
            *[set(data[x].date) for x in data]
        )
        idx = list(idx)
        idx.sort()

        for key in data:
            df = data[key].set_index('date').loc[idx]

            if isinstance(periods, int):
                df = df.iloc[::periods]
            elif isinstance(periods, Iterable):
                df = df[df.index.isin(periods)]


            rets = QuoteMedia._ohlc_returns(df, return_type, periods==1)
                
            rets.name = key
            srs_list.append(rets)

        out = pd.concat(srs_list, axis=1)
        return out

class Fred(Shelf):

    """
    Pull data from FRED (Federal Reserve Economic Data). Note that `unlike` ``QuoteMedia.get``, 
    ``FRED.get`` does not require an api_key.
    """

    endpoint = 'fred'

    @classmethod
    def pull(cls, symbols: Union[str, List[str]], convert_to_trading_days=False, **kwargs) -> None:

        codes = [symbols] if isinstance(symbols, str) else symbols

        if not codes:
            return
        
        data = get_data_fred(codes, **kwargs)

        with shelve.open(cls._fp(), writeback=True) as db:
            for c in data.columns:
                db[c] = data[c]

    def get_metadata(ids: Union[str, List[str]], api_key=None, raise_http_exception=True) -> pd.DataFrame:
        return get_metadata_fred(ids, api_key, raise_http_exception)

    def get_change(ids: Union[str, List[str]], overwrite=False, **kwargs) -> pd.DataFrame:
        data = Fred.get(ids, overwrite, **kwargs)
        metadata = Fred.get_metadata(ids)
        symb = metadata.loc[~metadata['units'].str.contains('Change'), 'id'].values

        for s in symb:
            data[s] = np.log(data[s]/data[s].shift())

        df = pd.DataFrame(data)
        df = df.rename(metadata['title'], axis=1)

        return df

class FamaFrench(Shelf):

    endpoint = 'famafrench'

    @classmethod
    def pull(cls, codes: Union[str, List[str]], key=0, **kwargs) -> pd.DataFrame:
        if isinstance(codes, str):
            codes = [codes]
        with shelve.open(cls._fp()) as db:
            for i in codes:
                db[i] = get_data_famafrench(i, key, **kwargs)

def get_data_fred(codes, **kwargs) -> pd.DataFrame:
    try:
        data = pdr.get_data_fred(codes, **kwargs)
    except RemoteDataError:
        if isinstance(codes, str):
            raise ValueError("'{}' is an invalid FRED code".format(codes))

        for c in codes:
            try:
                pdr.get_data_fred(c, **kwargs)
            except:
                raise ValueError("'{}' is an invalid FRED code".format(c))

    return data

def get_data_famafrench(code: str, key=0, **kwargs) -> pd.DataFrame:


    data_dict = pdr.get_data_famafrench(code, **kwargs)
    data = data_dict[key]

    if key=='DESCR':
        return data

    data /= 100

    if isinstance(data.index, pd.PeriodIndex):
        data.index = data.index.to_timestamp()

    return data

def get_data_quotemedia(tickers: Union[str, List[str]], api_key=None, start=None, end=None) -> pd.DataFrame:

    if not isinstance(api_key, str):
        api_key = ApiConfig.quotemedia

    
    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(start, dt.datetime):

        if not isinstance(end, dt.datetime):
            end = dt.datetime.today()
        
        dates = [
            x.isoformat() for x in pd.date_range(start=start, end=end, freq='d')
        ]
        data = quandl.get_table('QUOTEMEDIA/PRICES', ticker=tickers, date=dates, paginate=True, api_key=api_key)
    else:
        data = quandl.get_table('QUOTEMEDIA/PRICES', ticker=tickers, paginate=True, api_key=api_key)
        
    return data

def get_metadata_quotemedia(api_key=None):

    if not isinstance(api_key, str):
        api_key = ApiConfig.quotemedia

    #api call
    data = quandl.get_table('QUOTEMEDIA/TICKERS', paginate=True, api_key=api_key)

    #drop test data
    data = data.drop(data[data['company_name'].str.lower().str.contains('test')].index, axis=0)
    return data

def get_metadata_famafrench() -> List[str]:
    out = pdr.famafrench.FamaFrenchReader('').get_available_datasets()

    return out

def get_metadata_fred(ids: Union[str, List[str]], api_key=None, raise_http_exception=True):
    if isinstance(ids, str):
        ids = [ids]
        return_srs = True
    else:
        return_srs = False
    if api_key is None:
        api_key = ApiConfig.Fred
    srs_list = []
    endpoint = 'https://api.stlouisfed.org/fred/series'
    params = {'api_key': api_key, 'file_type': 'json'}
    for i in ids:
        params['series_id'] = i
        response = requests.get(endpoint, params=params)
        status = response.status_code
        json_out = json.loads(response.text)

        if status!=200:
            if raise_http_exception:
                err_msg = json_out['error_message']
                raise Exception('HTTP Error {}: {}'.format(status, err_msg))
            else:
                continue

        else:
            data_out = json_out['seriess'][0]
            srs = pd.Series(data_out, name=i)
            srs_list.append(srs)

    if return_srs:
        return srs_list.pop()

    return pd.concat(srs_list, axis=1).T
            
def convert_to_trading_days(index, frequency='d', exchange='NYSE'):
    """
    Transforms an iterable of timestamps to their closest trading period on or after
    the timestamp.
    
    This is useful for reconciling market data with non-market data e.g. economic data 
    that has timestamps which fall outside of regular trading hours.

    Args:
        index (Iterable of timestamps): The index to be converted.
        frequency (str): This affects the universe of potential new timestamps. For example,
        setting this to 'w' would limit the new timestamps to one per week.
        exchange (str): The exchange associated with the trading calendar.

    Returns:
        An iterable of transformed timestamps.
    
    """
    end_date = dt.datetime.today()
    start_date = min(index)
    nyse = mcal.get_calendar(exchange)
    sched = nyse.schedule(start_date=start_date, end_date=end_date)
    idx = sched[~sched.index.to_period(frequency).duplicated()].index
    new_index = index.map(lambda x: idx[idx>=x][0])
    return new_index


