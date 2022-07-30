import webbrowser
import streamlit as st
import os
from bt_view.data import Eod
from bt_view import open_docs


st.set_page_config(layout='wide')

def make_cache_key(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

def setup():
    script_loc = os.path.dirname(__file__)
    dpath = os.path.join(script_loc, 'data')
    subdirs = os.listdir(dpath)
    newdirs = ['strats', 'backtests']

    for n in newdirs:
        if n not in subdirs:
            os.mkdir(
                os.path.join(dpath, n)
            )

    make_cache_key('result', None)
    make_cache_key('backtests', [])
    make_cache_key('quotemedia', Eod())



if __name__ == '__main__':
    st.title('Backtester')
    st.markdown(
        '''
        The purpose of this application is to provide quick 
         data and visuals for backtests with a simple interface. 
         There are two pages: **Input** and **Output**. You can view these pages by opening
          the sidebar panel on the top left of the screen. The sidebar panel may contain
           supplementary information on particular pages.
        '''
    )

    st.markdown(
        '''
        The backtester is in early stage development. You are encouraged
         to open an issue on the project's GitHub page with any ideas for new features
         or bug issues. 
         '''
    )
    st.subheader('Input')

    st.markdown(
        '''
        Conceptually, a backtest is the simulation of a trading strategy
         on a particular set of historical asset prices. Backtests are added one
          by one to the backtest list (see the side panel). Pressing '''
    )

    st.sidebar.button('Module Documentation', on_click=open_docs)

    setup()