#%%
import numpy as np
import streamlit as st
from bt import Backtest, Strategy
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pickle
import bt
from typing import List


from .stream.bases import Pandas, Bt
from .stream.elements import BokehPlot
from bokeh.layouts import layout

from .elements import QuandlImport

class PageBase(Pandas, Bt):

    @property
    def hdf(self):
        if 'hdf' not in self.data:
            self.data['hdf'] = {}
        return self.data['hdf']

    def from_hdf_path(self, container):

        path = container.text_input('Path to HDF', key='_'.join(self.identifier, 'hdf', 'path'))
        field = container.text_input('Path to Dataset', key='_'.join(self.identifier, 'hdf', 'field'))

        if all((path, field)):
            return pd.read_hdf(path, field)
        
    def data_frame_to_hdf(self, container):

        key = container.selectbox('Select Dataset', self.datasets.keys(), key='_'.join(self.identifier, 'hdf_data'))
        path = container.text_input('Path to HDF', key='_'.join(self.identifier, 'hdf_path'))
        field = container.text_input('Path to Dataset', key='_'.join(self.identifier, 'hdf_field'))

        if all(key, path, field):
            self.datasets[key].to_hdf(path, field)
            container.success('{} saved to {}:{}'.format(key, path, field))

    def backtest_to_pickle(self, container, key=None):
        key = key or self.identifier
        path = container.text_input('Path to Save Pickle File', key=key)


    def backtest_from_pickle(self, container, label='Upload Backtest', key=None):
        f = st.file_uploader(label, 'pkl', False, key)
        if f is not None:
            bk = pickle.loads(f)
            if isinstance(bk, Backtest):
                self.backtests[bk.name] = bk
                container.success('Successfully read Backtest {}'.format(bk.name))

            elif isinstance(bk, Strategy):
                self.strats[bk.name] = bk
                container.success('Successfully read Strategy {}'.format(bk.name))

            else:
                raise IOError('Did not recognize object type')

    
    def io_panel(self, container):

        io_col, field_col = container.columns(2)

        io_selection = io_col.selectbox('I/O', ['Input', 'Output'])
        
        if io_selection=='Input':
            
            return self.backtest_from_pickle(field_col, 'Upload Backtest or Strategy')

        elif io_selection=='Pickle':
            return self.backtest_to_pickle(field_col)

    def selection_panel(self, container):

        bins = {
            'DataFrame': self.datasets,
            'Strategy': self.strats,
            'Backtest': self.backtests
        }

        stype = container.radio('Type', bins)

        data_bin = bins[stype]

        return data_bin[
            container.selectbox(stype, data_bin.keys())
        ]

class ReferencePage(Pandas):

    def __call__(self, *args, **kwargs):
        
        quandl_lookup = QuandlImport()
        prices = quandl_lookup()

        if prices.empty:
            st.stop()


        ttips = [('Price', '$@$name'), ('Date', '@index')]
        bok_plot = BokehPlot(prices, ttips=ttips)
        fig = bok_plot.line(x='Date')
        fig = bok_plot.format(fig, 'Date', 'Price')
        st.bokeh_chart(fig)

class BacktestLandingPage(PageBase):


    @property
    def backtest_fields(self):
        all_fields = ['prices', 'herfindahl_index', 'turnover']
        field_options = {}

        def attr_finder(bk, attr):
            try:
                field = getattr(bk, attr)
            except AttributeError:
                field = getattr(bk.strategy, attr)
            
            return field

        field_dict = lambda res, attr: {bk.name: attr_finder(bk, attr) for bk in res.backtest_list}
        for f in all_fields:
            field_options[f] = lambda res: pd.DataFrame(field_dict(res, f))

        return field_options

 

    def prices(self, *bkts):
        res = bt.run(*bkts)
        return res.prices

    def herfindahl_index(self, *bkts):
        df = pd.concat(
            [pd.Series(bk.herfindahl_index, name=bk.name) for bk in bkts],
            axis=1
        )
        return df

    def turnover(self, *bkts):
        df = pd.concat(
            [pd.Series(bk.turnover*100, name=bk.name) for bk in bkts],
            axis=1
        )
        return df


    def metric_groups(self, stats: pd.DataFrame, main_bkt_name: str, bench_bkt_name: str, container):

        groups = dict(
            All=None,
            Returns=self.return_metrics,
            Drawdown=self.drawdown_metrics,
            Yearly=self.yearly_metrics,
            Monthly=self.monthly_metrics,
            Daily=self.daily_metrics
        )

        group_selection_key = container.selectbox('Metrics', groups.keys())
        group_selection = groups[group_selection_key]


        if group_selection is None:
            new_stats = stats.drop(['start', 'end'], axis=0, errors='ignore')
            container.table(new_stats[[main_bkt_name, bench_bkt_name]])
        else:
            new_stats = stats.loc[group_selection]
            self.pretty_metrics(new_stats, container, bench_bkt_name)



    def __call__(self, **kwargs):
        strat_col, benchmark_col = st.columns(2)
        stats_col, plots_col = st.columns((1,2))

        bks_dict = {b.name: b for b in self.data['backtests']}
        strat_name = strat_col.selectbox('Backtest', bks_dict, key='s1')

        if strat_name is None:
            st.stop()

        bench_opts = [x for x in bks_dict if x!=strat_name]        
        bench_name = benchmark_col.selectbox('Benchmark', bench_opts, key='s2')

        if any([x is None for x in (strat_name, bench_name)]):
            st.markdown('Please select a Backtest and Benchmark to continue')
            st.stop()

        bench = bks_dict[bench_name]
        strat = bks_dict[strat_name]

        if all([x is None]for x in (strat,bench)):
            res = bt.run(strat, bench)
        else:
            st.stop()

        stats_col.subheader('Metrics')
        self.metric_groups(res.stats, strat.name, bench.name, stats_col)

        ttips = [('Price', '$@$name'), ('Date', '@index')]
        plots_col.subheader('Time Series')
        first_plot_opts = {'Price ($)': self.prices, 'Turnover (%)': self.turnover, 'Herfindahl Index': self.herfindahl_index}
        first_plot_key = plots_col.selectbox('Fields', first_plot_opts)
        func = first_plot_opts[first_plot_key]

        tseries = func(strat, bench)
        bok = BokehPlot(tseries, ttips=ttips, )
        bok_fig = bok.line('index', [strat.name, bench.name])
        bok_fig = bok.format(bok_fig, 'Date', first_plot_key)
        plots_col.bokeh_chart(bok_fig)

        sec_pos = BokehPlot(strat.positions)
        sec_pos_fig = sec_pos.line('index', [col for col in strat.positions])
        sec_pos_fig = sec_pos.format(sec_pos_fig, 'Date', 'Shares', '{} Positions'.format(strat.name))
        plots_col.bokeh_chart(sec_pos_fig)

        sec_wts = BokehPlot(strat.security_weights)
        sec_wts_fig = sec_wts.line('index', [col for col in strat.security_weights])
        sec_wts_fig = sec_wts.format(sec_wts_fig, 'Date', 'Weight', '{} Security Weights'.format(strat.name))
        plots_col.bokeh_chart(sec_wts_fig)

class RollingOLSPage(PageBase):

    def info_criterion_plot(self, res):
        bottom_left_metrics = ['aic', 'bic']

    
    def result_summary(self, res):
        metrics = rolling_regression_metrics(res)
        bottom_left_metrics = ['aic', 'bic']
        gof_metrics = ['rsquared_adj', 'f_pvalue']
        bottom_right_metrics = ['llf']

        bk_plot = BokehPlot(metrics)
        params = bk_plot.line(y=[x for x in metrics if 'param' in x])
        pvals = bk_plot.line(y=[x for x in metrics if 'pval' in x])
        gof = bk_plot.line(y=gof_metrics)

        params = bk_plot.format(params, 'Time', 'Value', 'Parameter Estimates')
        pvals = bk_plot.format(pvals, 'Time', 'Value', 'Parameter Estimate P-Values')
        gof = bk_plot.format(gof, 'Time', 'Value', 'Quality Metrics')

        fig = layout([[params], [pvals, gof]])
        return fig

    def __call__(self, *args, **kwargs):
        bks = self.data['backtests']
        bk_dict = {x.name: x for x in bks}
        bk_selection_name = st.selectbox('Backtest', bk_dict)
        if not bk_selection_name:
            st.stop()
        exog_options = list(bk_dict.keys())
        exog_options.remove(bk_selection_name)
        bk_exog = st.multiselect('Factor Selection', exog_options)

        bk_selection = bk_dict[bk_selection_name]
        endog = np.log(bk_selection.strategy.prices/bk_selection.strategy.prices.shift(1)).dropna().iloc[1:]
        if not bk_exog:
            exog = np.log(bk_selection.data/bk_selection.data.shift(1)).dropna()
        else:
            exog_bks = [bk_dict[x] for x in bk_exog]
            exog_df = pd.concat([pd.Series(b.strategy.prices, name=b.name) for b in exog_bks], axis=1)
            exog = np.log(exog_df/exog_df.shift(1)).dropna()
            
        exog = sm.add_constant(exog)

        window = st.number_input('Window Length', 1, len(endog)-1, int(len(endog)/5))
        mod = RollingOLS(endog, exog, window, min_nobs=window)
        res = mod.fit()
        fig = self.result_summary(res)
        st.bokeh_chart(fig)



def rolling_regression_metrics(res):

    srs_attrs = [
        'aic',
        'bic',
        'llf',
        'rsquared',
        'rsquared_adj',
        'fvalue',
        'f_pvalue',
        'mse_model',
        'mse_resid',
        'mse_total',
    ]

    metrics = pd.concat(
            {
                a: getattr(res, a) for a in srs_attrs
            },
        axis=1)

    xnames = res.model.data.xnames
    make_cols = lambda prefix: ['{}_{}'.format(prefix, x) for x in xnames]
    bse = res.bse
    bse.columns = make_cols('bse')
    pvals = pd.DataFrame(res.pvalues, index=bse.index, columns=make_cols('pval'))
    tvals = res.tvalues
    tvals.columns = make_cols('tval')
    params = res.params
    params.columns = make_cols('param')

    df = pd.concat([params, pvals, metrics, tvals, bse], axis=1)
    return df


def reset_app():
    del st.session_state['pages']

def app(*bks):
    st.set_page_config(layout='wide')


    if 'pages' not in st.session_state:
        res = bt.run(*bks)
        data = {'backtests': res.backtest_list}

        pages = {
            'Main': BacktestLandingPage('Main', data=data),
            'Price Lookup': ReferencePage('Price Lookup', data=data),
            'Regression': RollingOLSPage('Regression', data=data)
        }
        st.session_state['pages'] = pages
    
    pages = st.session_state['pages']

    p_name = st.sidebar.selectbox('Page', pages)
    st.sidebar.button('Reset', on_click=reset_app)
    p = pages[p_name]
    p.setup()
    p()
