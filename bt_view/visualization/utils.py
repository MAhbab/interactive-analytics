'''Utility functions to aid in generating visuals'''
from enum import Enum
import pandas as pd
import numpy as np
import bokeh

from bokeh.plotting import figure, Figure
from bokeh.models import HoverTool, ColumnDataSource, DataTable, TableColumn
from itertools import cycle
from scipy.stats import probplot
from statsmodels.tsa.stattools import acf
from bt.backtest import Result
from typing import List, Union

HEIGHT = 400
WIDTH = 600
TOOLBAR_LOC = 'below'
SIZING_MODE = 'stretch_both'

class FfnMetric(Enum):
    pass

class Returns(FfnMetric):
    TOTAL = 'total_return'
    CAGR = 'cagr'
    MONTH_TO_DATE = 'mtd'
    THREE_MONTH = 'three_month'
    SIX_MONTH = 'six_month'
    YEAR_TO_DATE = 'ytd'
    ONE_YEAR = 'one_year'
    THREE_YEAR = 'three_year'
    FIVE_YEAR = 'five_year'
    TEN_YEAR = 'ten_year'
    SINCE_INCEPTION = 'incep'

class Drawdown(FfnMetric):
    MAX = 'max_drawdown'
    AVERAGE = 'avg_drawdown'
    AVERAGE_LENGTH_DAYS = 'avg_drawdown_days'

class Daily(FfnMetric):
    MEAN = 'daily_mean'
    VOLATILITY = 'daily_volatility'
    SKEW = 'daily_skew'
    KURTOSIS = 'daily_kurt'
    SHARPE = 'daily_sharpe'
    SORTINO = 'daily_sortino'
    BEST = 'best_day'
    WORST = 'worst_day'

class Monthly(FfnMetric):
    MEAN = 'monthly_mean'
    VOLATILITY = 'monthly_volatility'
    SKEW = 'monthly_skew'
    KURTOSIS = 'monthly_kurt'
    SHARPE = 'monthly_sharpe'
    SORTINO = 'monthly_sortino'
    BEST = 'best_month'
    WORST = 'worst_month'

class Yearly(FfnMetric):
    MEAN = 'yearly_mean'
    VOLATILITY = 'yearly_volatility'
    SKEW = 'yearly_skew'
    KURTOSIS = 'yearly_kurt'
    SHARPE = 'yearly_sharpe'
    SORTINO = 'yearly_sortino'
    BEST = 'best_year'
    WORST = 'worst_year'

class Sharpe(FfnMetric):
    DAILY = 'daily_sharpe'
    MONTHLY = 'monthly_sharpe'
    YEARLY = 'yearly_sharpe'

class Sortino(FfnMetric):
    DAILY = 'daily_sortino'
    MONTHLY = 'monthly_sortino'
    YEARLY = 'yearly_sortino'


def default_figure(datetime_x=False, ttips=None) -> Figure:
    """
    The default bokeh figure used to make visuals.
    
    Args:
        datetime_x: Set as true if x-axis contains datetime values.
        ttips: Tooltips to include at the figure level. See 
            https://docs.bokeh.org/en/latest/docs/user_guide/tools.html#basic-tooltips
            for furher information.
        
    Returns:
        A bokeh figure
        
    """
    fig_kwargs = {
        'height': HEIGHT,
        'width': WIDTH,
        'toolbar_location': TOOLBAR_LOC,
        'sizing_mode': SIZING_MODE,
        'tools': "pan,wheel_zoom,box_zoom,tap,box_select,poly_select,reset,save"
    }
    if datetime_x:
        fig_kwargs['x_axis_type'] = 'datetime'
    fig = figure(**fig_kwargs)
    if ttips is not None:
        hover = HoverTool(tooltips=ttips, mode='vline')
        fig.add_tools(hover)
    
    return fig

def bkformat(figure: Figure, xlabel: str = None, ylabel: str= None, title: str = None):
    """
    Helper function to perform basic formatting on bokeh figures.
    Specifically, this adds a title, axis labels, and an interactive
    legend.
    
    Args:
        figure: The figure to format.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Figure title
        
    Returns:
        Formatted figure.
        
    """
    figure.legend.location = 'top_left'
    figure.legend.click_policy = 'hide'

    if xlabel is not None:
        figure.xaxis.axis_label = xlabel
    if ylabel is not None:
        figure.yaxis.axis_label = ylabel
    figure.axis.axis_label_text_font_size = '12pt'
    
    if title is not None:
        figure.title.text = title
        figure.title.text_font_size = '16pt'
    return figure

def time_series(
    dframe: pd.DataFrame,
    color='viridis',
    xlabel: str = None,
    ylabel: str = None,
    title: str = None
) -> Figure:

    """
    Generate an interactive time series plot from a DataFrame. Requires that
    the index be a pandas DateTimeIndex.

    The generated plot can be viewed in an HTML document using :func: bokeh.plotting.show
    or in a Streamlit app using :func: streamlit.bokeh_chart. For the latter, see
    https://docs.streamlit.io/library/api-reference/charts/st.bokeh_chart for more info.

    NOTE: Streamlit requires bokeh version 2.4.1 to view charts in app.
    
    Args:
        dframe: Data to be plotted.
        color: Color palette to use. See https://docs.bokeh.org/en/latest/docs/reference/palettes.html
        for more information.
        xlabel: The plot's x-axis label.
        ylabel: The plot's y-axis label.
        title: The plot's title.
        prefix: 
        
    Returns:
        A figure viewable in HTML or a Streamlit app.
    """

    #pre process data
    COLS = list(dframe.columns)

    ttips = [('Name', '@name'), ('Value', '@value{0.00a}'), ('Date', '@date_str')]
    COLORS = cycle(
        getattr(bokeh.palettes, color, bokeh.palettes.viridis)(len(COLS)+1)
    )

    fig = default_figure(True, ttips)


    for c in COLS:
        df = dframe[c]

        #in case dframe has multi-index columns
        if isinstance(c, tuple):
            name = ':'.join(c)
        else:
            name = c
        df.name = 'value'
        df.index.name = 'date'
        df = df.reset_index()
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df['name'] = name

        src = ColumnDataSource(df)
        clr = next(COLORS)
        fig.line(y='value', x='date', source=src, legend_label=name, color=clr, name=name)

    fig_formatted = bkformat(fig, xlabel, ylabel, title)

    return fig_formatted

def hist(dframe: pd.DataFrame, bins: int, color: str = 'viridis'):

    #helper function that returns bar dimensions
    #to plot returns distribution
    def bquad(srs, bins):
        bounds_dict = {}
        try:
            rbins = pd.cut(srs, np.linspace(srs.min(), srs.max(), bins))
        except ValueError:
            raise Exception('Error generating bins. Try decrease the bin count or setting a longer timeframe')
        rgrp = rbins.groupby(rbins).count()
        bounds_dict['top'] = [x for x in rgrp]
        bounds_dict['left'] = [x.left for x in rgrp.index]
        bounds_dict['right'] = [x.right for x in rgrp.index]
        bounds_dict['bottom'] = [0]*len(rgrp)
        return bounds_dict

    ncols = len(dframe.columns)
    COLORS = cycle(getattr(bokeh.palettes, color, bokeh.palettes.viridis)(ncols+1))
    ttips = [('Name', '$name'), ('Count', '@top'), ('Value', '$x{0,0.00}')]


    myfig = default_figure(ttips=ttips)
    for k in dframe.columns:
        kw = bquad(dframe[k], bins)
        myfig.quad(
            top=kw['top'],
            bottom=kw['bottom'],
            left=kw['left'],
            right=kw['right'],
            legend_label=k, 
            color=next(COLORS),
            name=k
            )
    return myfig

def prob_plot(rets, fig=None, **kwargs):
    data, params = probplot(rets, **kwargs)
    fitted = (params[0]*data[0]+params[1])

    if not fig:
        fig = default_figure()

    if isinstance(rets, pd.Series):
        name = rets.name 
    else:
        name = 'Returns'
    
    fitted_name = ' '.join(['Fitted', name])

    fig.scatter(x=data[0], y=data[1], legend_label=name, name=name)
    fig.line(x=data[0], y=fitted, legend_label=fitted_name, name=fitted_name)
    return fig 

def acf_plot(rets, fig=None, adjusted=False, alpha=0.05, nlags=None):
        
    data, conf = acf(rets, adjusted=adjusted, alpha=alpha, nlags=nlags)
    xrange = [x for x in range(len(data))]
    if not fig:
        fig = default_figure()

    if isinstance(rets, pd.Series):
        name = rets.name 
    else:
        name = 'ACF'
    
    conf_name = ' '.join([name, '{}%'.format(int((1-alpha)*100)),'Confidence Interval'])

    fig.line(x=xrange, y=data, legend_label=name)
    fig.varea(x=xrange, y1=conf[:, 0], y2=conf[:, 1], alpha=0.2, legend_label=conf_name)
    return fig 

def ffn_stats(stats: pd.DataFrame, *metrics: Union[FfnMetric, str]) -> DataTable:

    '''
    Generate time series financial metrics in a Bokeh object.
    
    Args:
        res: Backtest results.
        metrics: Specify metrics to display. If this is None,
         all metrics will be displayed.
        
        Returns:
            A table of financial time series metrics.
    '''

    float_ids = stats.index.drop(['start', 'end'], errors='ignore')
    stats.loc[float_ids] = stats.loc[float_ids].astype(float).round(3)
    stats.loc[['start', 'end']] = stats.loc[['start', 'end']].astype(str)

    if metrics:
        all_metrics = stats.index
        if isinstance(metrics[0], FfnMetric):
            metric_strings = ['start', 'end'] + [x.value for x in metrics]
        elif isinstance(metrics[0], str):
            metric_strings = ['start', 'end'] + [x for x in metrics]
        else:
            raise TypeError('Invalid type for metric: {}'.format(type(metrics[0])))
        chosen_metrics = all_metrics.intersection(metric_strings)
        stats = stats.loc[chosen_metrics]

    new_table = stats.copy()
    new_table.index = new_table.index.str.replace('_', ' ').str.title()
    new_table = new_table.dropna()
    
    metrics = ColumnDataSource(new_table)
    cols = [TableColumn(field=x, title=x.title()) for x in metrics.data.keys()]
    table = DataTable(source=metrics, columns=cols)
    return table












