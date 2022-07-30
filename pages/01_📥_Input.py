import pandas as pd
import streamlit as st
import datetime as dt
import pickle
import numpy as np
import bt
from bt.backtest import Result
import string, random

from bt_view.visualization import configure
from bt_view.data import Eod
from enum import Enum
from typing import List, Tuple

class InputMethod(Enum):
    TEST = 'Specify Portfolio with Test Data'
    PICKLE = 'Pickle File'
    PORTFOLIO = 'Specify Portfolio'
    CSV = 'From Weights'

    def input_from_pickle() -> bt.Backtest:
        return load_from_pickle()

    def input_portfolio() -> Tuple[bt.Strategy, List[str]]:
        algos = configure()
        tickers = backtest_from_quotemedia()
        name = st.text_input('Name this strategy', 'My Portfolio Strategy')
        strat = bt.Strategy(name, algos)
        return strat, tickers

    def input_portfolio_test() -> bt.Backtest:
        start_date = dt.datetime(2015,1,1)
        end_date = dt.datetime(2020,1,1)
        algos = configure()
        suffix = random.choices(string.ascii_uppercase, k=4)
        strat = bt.Strategy('Test Strategy {}'.format(''.join(suffix)), algos)
        data = generate_data(start_date, end_date, 5)
        return bt.Backtest(strat, data)

    def input_weight_dataframe() -> Tuple[bt.Strategy, List[str]]:
        help_msg = "Invalid columns values (stock tickers) will be removed"
        f = st.file_uploader('Upload Weights Dataframe', 'csv', False, help=help_msg)
        if f is None:
            st.stop()

        name = st.text_input('Name this strategy', 'My Custom Weights Strategy')
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        wts = validated_weights(df)
        tickers = list(wts.columns)
        
        strat = bt.Strategy(name, [
            bt.algos.WeighTarget(wts),
            bt.algos.Rebalance()
        ])

        return strat, tickers


def load_from_pickle():

    f = st.file_uploader('Upload Results as pickle file', 'pkl')

    if f is None:
        st.stop()

    data = f.read()
    bk:bt.Backtest = pickle.loads(data)
    if not isinstance(bk, bt.Backtest):
        st.error('ERROR: Uploaded file is not a valid backtest.')
        return None
    return bk

def generate_data(start_date, end_date, num_securities):
    idx = pd.date_range(start_date, end_date, freq='d')
    daily_vol = 0.005
    dim = (len(idx), num_securities)
    df = (1+pd.DataFrame(np.random.normal(0, daily_vol, dim), index=idx)).cumprod()*100
    df.columns = [str(x) for x in df.columns]
    return df

def backtest_from_quotemedia() -> Result:
    src: Eod = st.session_state['quotemedia']

    h = 'Display company name and primary exchange'
    vb = st.checkbox('Verbose ticker names', True, help=h)

    name = 'Portfolio Assets'
    tickers = src.prices_interface_condensed(vb, name, name)

    return tickers

def validated_weights(weights: pd.DataFrame) -> pd.DataFrame:
    src: Eod = st.session_state['quotemedia']
    tickers = weights.columns
    unv = src.meta_data['ticker'].values
    valid_tickers = tickers.intersection(unv)

    if len(valid_tickers) != len(tickers):
        invalid = tickers.difference(valid_tickers)
        st.write('Invalid tickers detected: {}'.format(invalid))
        weights = weights.drop(invalid, axis=1)
    
    return weights


def read_results():

    override = False

    selection = st.selectbox(
        'Input method',
        InputMethod._member_names_,
        0,
        lambda x: InputMethod._member_map_[x].value,
        None,
        'Choose how you would like to backtest'
    )

    tickers = []
    strat = None

    if selection == InputMethod.PICKLE.name:
        bk = InputMethod.input_from_pickle()

    elif selection == InputMethod.PORTFOLIO.name:
        strat, tickers = InputMethod.input_portfolio()
        override = True

    elif selection == InputMethod.CSV.name:
        strat, tickers = InputMethod.input_weight_dataframe()

    elif selection == InputMethod.TEST.name:
        bk = InputMethod.input_portfolio_test()

    else:
        raise Exception("Input method is invalid")

    if (st.button('Save')) or (override):
        if isinstance(strat, bt.Strategy):
            with st.spinner('Fetching stock data..'):
                src: Eod = st.session_state['quotemedia']
                data = src.get(tickers).to_price_series()
                st.dataframe(data)
                bk = bt.Backtest(strat, data)
        return bk

def backtest_callback():
    bks = st.session_state['backtests']
    with st.spinner('Running backtests..'):
        try:
            res = bt.run(*bks)
            st.session_state['result'] = res
            st.success('Results saved!')
        except:
            st.error('There was an error running the backtests')
            bt.run(*bks) #run backtests again to show error message
        
        finally:
            st.session_state['backtests'] = []



if __name__ == '__main__':

    st.sidebar.markdown('Backtests in List: {}'.format(len(st.session_state['backtests'])))
    st.sidebar.markdown('Backtest Results Loaded: {}'.format(isinstance(st.session_state['result'], Result)))


    bk = read_results()

    if isinstance(bk, bt.Backtest):
        st.session_state['backtests'].append(bk)
        st.success('Backtest saved!')


    st.subheader('Backtests')
    st.write({x.name: x for x in st.session_state['backtests']})
    st.button('Run', on_click=backtest_callback)
