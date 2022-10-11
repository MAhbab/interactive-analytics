#%%
import unittest
import pandas as pd
import numpy as np
import datetime as dt
import btplus as btp


def generate_ohlcv(mu=0.005, sig=0.04, n=252, seed=70, name='TKR'):
    np.random.seed(seed)
    bounded_price_func = lambda x: np.random.uniform(x['adj_low'], x['adj_high'])
    

    idx = pd.date_range(start=dt.datetime(2000,1,1), periods=n, freq='d')
    low = np.random.normal(mu, sig, n)
    df = pd.DataFrame({'adj_low': low}, index=idx)
    df += 1
    df.iloc[0]  = 100
    df = df.cumprod()

    df['adj_high'] = df['adj_low']*np.random.uniform(1.01, 1.3, n)
    df['adj_open'] = df.apply(bounded_price_func, axis=1)
    df['adj_close'] = df.apply(bounded_price_func, axis=1)
    df['adj_volume'] = np.log(df['adj_high']/df['adj_low'])*(10**9)
    df['ticker'] = name
    df.reset_index(inplace=True)
    df.rename({'index': 'date'}, inplace=True, axis=1)

    return df


def replacement_quotemedia_get_method(tickers, overwrite=None):
    if isinstance(tickers, str):
        tickers = [tickers]

    out = {t: generate_ohlcv(name=t, seed=abs(hash(t))%(2**31)) for t in tickers}
    return out

btp.data.QuoteMedia.get = replacement_quotemedia_get_method

class TestQuoteMedia(unittest.TestCase):

    def test_period_returns_c2c_single_period(self):
        tkr = 'YEET'
        data_dict = btp.data.QuoteMedia.get(tkr)
        data = data_dict[tkr].set_index('date')
        rets = btp.data.QuoteMedia.period_returns(return_type='c2c', periods=1, tickers=tkr)
        
        date = dt.datetime(2000,3,2)
        prior_date = date - dt.timedelta(days=1)

        expected = np.log(data.loc[date, 'adj_close']/data.loc[prior_date, 'adj_close'])
        actual = rets.loc[date, tkr]

        self.assertEqual(expected, actual)

    def test_period_returns_c2c_multi_period(self):
        tkr = 'YEET'
        p = 5
        data = btp.data.QuoteMedia.get(tkr)[tkr].set_index('date')
        rets = btp.data.QuoteMedia.period_returns(tkr, return_type='c2c', periods=p)

        date = rets.index[15]
        prior_date = date - dt.timedelta(days=p)

        expected = np.log(data.loc[date, 'adj_close']/data.loc[prior_date, 'adj_close'])
        actual = rets.loc[date, tkr]

        self.assertEqual(expected, actual)

    def test_period_returns_c2o(self):
        tkr = 'YEET'
        data = btp.data.QuoteMedia.get(tkr)[tkr].set_index('date')
        rets = btp.data.QuoteMedia.period_returns(tkr, 'c2o', 1)

        date = dt.datetime(2000,6,5)
        prior_date = date - dt.timedelta(days=1)

        expected = np.log(data.loc[date, 'adj_open']/data.loc[prior_date, 'adj_close'])
        actual = rets.loc[date, tkr]

        self.assertEqual(expected, actual)

    def test_period_returns_o2c_single_period(self):
        tkr = 'YEET'
        data = btp.data.QuoteMedia.get(tkr)[tkr].set_index('date')
        rets = btp.data.QuoteMedia.period_returns(tkr, 'o2c', 1)

        date = dt.datetime(2000,5,4)
        
        expected = np.log(data.loc[date, 'adj_close']/data.loc[date, 'adj_open'])
        actual = rets.loc[date, tkr]

        self.assertEqual(expected, actual)




if __name__ == '__main__':
    unittest.main()

