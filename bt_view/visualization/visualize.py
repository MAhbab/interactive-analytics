'''
Collection of objects to help visualize backtest results. Right now,
 all visualizations are made with bokeh, with more visualization libraries
  coming down the pike.
'''
from itertools import cycle
import pandas as pd

from enum import Enum

from pandas import DataFrame
from bt.backtest import Result
from bokeh.plotting import Figure, ColumnDataSource
from bokeh.models import HoverTool, Panel, Tabs, DataTable
from bokeh.palettes import viridis
import bokeh
from typing import Dict, Union
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.regression.rolling import RollingRegressionResults

import hiplot
import streamlit as st
from .utils import FfnMetric, ffn_stats, time_series, default_figure

class ResultVisual:

    class Attribute(Enum):
        PRICES = 'prices'
        SECURITY_PRICES = 'data'
        POSITIONS = 'positions'
        WEIGHTS = 'weights'
        SECURITY_WEIGHTS = 'security_weights'
        TRANSACTIONS = 'transactions'
        HERFINDAHL_INDEX = 'herfindahl'
        TURNOVER = 'turnover'
        STATS = 'stats'



    """
    Object to generate different visualization layouts 
    of :class: bt.backtest.Result objects.
    
    Args:
        results: Instance of :class: bt.backtest.Result
        
    """

    def __init__(self, results: Result) -> None:
        self.res = results

    @property
    def data(self) -> Dict[str, DataFrame]:
        """Convenience dictionary with aggregated backtest data"""
        r = self.res
        bkts = r.backtest_list
        data = {
            self.Attribute.PRICES: pd.DataFrame({x.name: x.strategy.prices for x in bkts}),
            self.Attribute.TURNOVER: pd.DataFrame({x.name: x.turnover for x in bkts}),
            self.Attribute.HERFINDAHL_INDEX: pd.DataFrame({x.name: x.herfindahl_index for x in bkts}),
            self.Attribute.SECURITY_WEIGHTS: pd.concat({x.name: x.security_weights for x in bkts}, axis=1),
            self.Attribute.WEIGHTS: pd.concat({x.name: x.weights for x in bkts}, axis=1),
            self.Attribute.POSITIONS: pd.concat({x.name: x.positions for x in bkts}, axis=1),
            # error getting transactions when using bt.algos.WeighTarget
            self.Attribute.TRANSACTIONS: pd.concat({x.name: x.strategy.get_transactions() for x in bkts}, axis=1),
            self.Attribute.SECURITY_PRICES: pd.concat({x.name: x.data for x in bkts}, axis=1),
            self.Attribute.STATS: r.stats
        }
        return data

    @property
    def stats(self):
        return self.res.stats

    def plot(self, df, **kwargs) -> Figure:
        fig = time_series(df, **kwargs)
        return fig

    @st.cache(hash_funcs={bokeh.document.Document: lambda x: None, bokeh.document.document.Document: lambda x: None}, allow_output_mutation=True)
    def paginated_plot(self, bk_name: str, *fields: Attribute):
        tabs = []
        invalid_fields = [
            self.Attribute.TRANSACTIONS,
            self.Attribute.STATS
        ]
        if not fields:
            fields = self.Attribute._member_map_.values()

        for f in fields:
            if f in invalid_fields:
                continue

            title_suffix = f.name.lower().replace('_', ' ').title()
            title = ' '.join([bk_name, title_suffix])
            df = self.data[f]
            p = self.plot(df, xlabel='Time', ylabel=title_suffix, title=title)
            tabs.append(
                Panel(child=p, title=title_suffix)
            )

        fig = Tabs(tabs=tabs)
        return fig

    def stats_table(self, *metrics: Union[FfnMetric, str]) -> DataTable:
        return ffn_stats(self.data[self.Attribute.STATS], *metrics)

    def stats_hiplot(self, *metrics: Union[FfnMetric, str]) -> None:
        stats = self.data[self.Attribute.STATS].drop(['start', 'end'], axis=0, errors='ignore')
        if metrics:
            if isinstance(metrics[0], FfnMetric):
                stats = stats[stats.index.isin([m.value for m in metrics])]
            elif isinstance(metrics[0], str):
                stats = stats[stats.index.isin(metrics)]
        xp = hiplot.Experiment.from_dataframe(stats.T.reset_index())
        xp.to_streamlit().display()

class RollingRegressionVisual:
    
    def __init__(self, results: RollingRegressionResults) -> None:
        self.res: RollingRegressionResults = results

    @property
    def data(self):
        params = self.res.params.copy()
        pvals = pd.DataFrame(self.res.pvalues, index=params.index, columns=params.columns)
        params['date_str'] = params.index.strftime('%Y-%m-%d')
        pvals.columns = [x+'_pval' for x in pvals.columns]
        other_metrics = pd.DataFrame({'aic': self.res.aic, 'rsquared_adj': self.res.rsquared_adj, 'bic': self.res.bic})
        combined = pd.concat([params, pvals, other_metrics], axis=1)
        return combined

    def plot_params(self, include_errors=False, pval_thresh=None):

        data = self.data
        params = self.res.params.copy()
        src = ColumnDataSource(data)
        lower = params - 2*self.res.bse
        lower.columns = [x+'_lower' for x in lower.columns]
        upper = params + 2*self.res.bse
        upper.columns = [x+'_upper' for x in upper.columns]
        errors = pd.concat([upper, lower], axis=1)
        esrc = ColumnDataSource(errors)

        colors = viridis(len(params.columns)+1)
        color_itr = cycle(colors)

        fig = default_figure(True)

        for c in params.columns:
            clr = next(color_itr)
            p1 = fig.line(source=src, y=c, x='index', color=clr, name=c, legend_label=c)
            ttips = [
                ('{}'.format(c), '@{}'.format(c)+'{0.00a}'),
                ('Date', '@date_str'),
                ('Pvalue', '@{}_pval'.format(c))
            ]
            ht = HoverTool(renderers=[p1], tooltips=ttips, mode='vline')
            fig.add_tools(ht)

            if include_errors:
                p2 = fig.varea(
                    source=esrc,
                    x='index',
                    y1='{}_upper'.format(c),
                    y2='{}_lower'.format(c),
                    color=clr,
                    alpha=0.3,
                    legend_label='{} 95% Conf. Interval'.format(c)
                )

        return fig



    
        
