"""
This module contains dynamic visualizations of several statistical and financial objects.

The preferred way to use many of the fuctions here is as the member functions of their associated
classes. For example, suppose you want to visualize a regression you performed using the ``Statsmodels``
package. There are two ways to do this.

Method 1 (Preferred):

>>> import btplus
>>> import statsmodels.api as sm
>>> model = sm.OLS(...)
>>> result = model.fit()
>>> result.panel(display=True)

Method 2:

>>> from btplus.visual import regression_panel
>>> import statsmodels.api as sm
>>> model = sm.OLS(...)
>>> result = model.fit()
>>> regression_panel(result, display=True)

"""
import pandas as pd
import numpy as np
import bokeh
import statsmodels.api as sm
import hiplot as hip

from bokeh.plotting import figure, Figure, show
from bokeh.models import HoverTool, ColumnDataSource, DataTable, TableColumn, CDSView, GroupFilter, Panel, Tabs
from bokeh.models.widgets import DateRangeSlider, DataTable, TableColumn, Div
from bokeh.layouts import column
from itertools import cycle
from scipy.stats import probplot
from statsmodels.tsa.stattools import acf
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.regression.rolling import RollingRegressionResults

HEIGHT = 400
WIDTH = 600
TOOLBAR_LOC = 'below'
SIZING_MODE = 'stretch_both'


def default_figure(datetime_x=False, ttips=None, **kwds) -> Figure:
    default_kwargs = {
        'height': HEIGHT,
        'width': WIDTH,
        'toolbar_location': TOOLBAR_LOC,
        'sizing_mode': SIZING_MODE,
        'tools': "pan,wheel_zoom,box_zoom,tap,box_select,poly_select,reset,save"
    }

    fig_kwargs = dict(default_kwargs, **kwds)
    if datetime_x:
        fig_kwargs['x_axis_type'] = 'datetime'
    fig = figure(**fig_kwargs)
    if ttips is not None:
        hover = HoverTool(tooltips=ttips, mode='vline')
        fig.add_tools(hover)
    
    return fig

def bkformat(figure: Figure, xlabel: str = None, ylabel: str= None, title: str = None):
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
    dframe,
    color='viridis',
    xlabel=None,
    ylabel=None,
    title=None,
    ttips_on=True,
    **fig_kwargs
):

    """
    Generate an interactive time series plot from a DataFrame. Requires that
    the index be a pandas DateTimeIndex.

    Args:
        dframe (DataFrame): Data to be plotted.
        color (str): Color palette to use. For more information, see here: https://docs.bokeh.org/en/latest/docs/reference/palettes.html
        xlabel (str, optional): The plot's x-axis label.
        ylabel (str, optional): The plot's y-axis label.
        title (str, optional): The plot's title.
        ttips_on (bool): Enables hover tooltips.
        
    Returns:
        An interactive time series figure.

    Examples:
        You can invoke this function as a member of ``DataFrame``. See below.
        This is the preferred way to call this function.

        >>> import btplus
        >>> import pandas as pd
        >>> from bokeh.plotting import output_notebook, show
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> ts = pd.DataFrame(...)
        >>> fig = ts.bokeh_time_series()
        >>> show(fig)
    """

    #pre process data
    if isinstance(dframe, pd.Series):
        dframe = pd.DataFrame(dframe)

    COLS = list(dframe.columns)

    if ttips_on:
        ttips = [('Name', '@name'), ('Value', '@value{0.00a}'), ('Date', '@date_str')]
    else:
        ttips = None
    COLORS = cycle(
        getattr(bokeh.palettes, color, bokeh.palettes.viridis)(len(COLS)+1)
    )

    fig = default_figure(True, ttips, **fig_kwargs)


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

def hist(dframe, bins, color='viridis', xlabel=None, ylabel=None, title=None):
    """
    Generate an interactive histogram plot from a DataFrame.

    Args:
        dframe (DataFrame): Data to be plotted.
        bins (int): Number of partitions to include in the plot.
        color: Color palette to use. For more information, see here: https://docs.bokeh.org/en/latest/docs/reference/palettes.html
        xlabel (str, optional): The plot's x-axis label.
        ylabel (str, optional): The plot's y-axis label.
        title (str, optional): The plot's title.
        
    Returns:
        An interactive histogram figure.

    Examples:
        You can invoke this function as a member of ``DataFrame``. See below.
        This is the preferred way to call this function.

        >>> import btplus
        >>> import pandas as pd
        >>> from bokeh.plotting import output_notebook, show
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> ts = pd.DataFrame(...)
        >>> fig = ts.bokeh_hist()
        >>> show(fig)
    """

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
    out = bkformat(myfig, xlabel, ylabel, title)

def prob_plot(rets, fig=None, **kwargs):

    if isinstance(rets, pd.Series):
        name = rets.name 
    else:
        name = 'Series'

    data, params = probplot(rets, **kwargs)
    fitted = (params[0]*data[0]+params[1])

    if not fig:
        fig = default_figure(x_axis_label='Z-score', y_axis_label='{}'.format(name), title='{} Probability Plot'.format(name))
    
    fitted_name = ' '.join(['Fitted', name])

    fig.scatter(x=data[0], y=data[1], legend_label=name, name=name)
    fig.line(x=data[0], y=fitted, legend_label=fitted_name, name=fitted_name)
    return fig 

def acf_plot(rets, fig=None, adjusted=False, alpha=0.05, nlags=None):

    if isinstance(rets, pd.Series):
        name = rets.name 
    else:
        name = 'ACF'
        
    data, conf = acf(rets, adjusted=adjusted, alpha=alpha, nlags=nlags)
    xrange = [x for x in range(len(data))]
    if not fig:
        fig = default_figure(y_axis_label=name, x_axis_label='Lag', title='{} Autocorrelation Function'.format(name))
    
    conf_name = ' '.join([name, '{}%'.format(int((1-alpha)*100)),'Confidence Interval'])

    fig.line(x=xrange, y=data, legend_label=name)
    fig.varea(x=xrange, y1=conf[:, 0], y2=conf[:, 1], alpha=0.2, legend_label=conf_name)
    return fig 

def regression_panel(res, display=True, exog_oos=None, endog_oos=None):
    """
    Display regression results as a dynamic panel. Optional out-of-sample data can be passed
    in to view predictions.
    
    Args:
        res (RegressionResults): The regression results to visualize.
        display (bool): If True, displays the panel. Otherwise, the panel is returned. The latter
        is useful for tabulating multiple results into the same graphic object.
        exog_oos (array_like, optional): An array/table of out-of-sample independent variable observations. Must have 
        the same number of columns as ``res.model.data.orig_exog``.
        endog_oos (array_like, optional): An array of observations corresponding to the predictions made with 
        ``exog_oos``.

    Returns:
        A dynamic bokeh layout

    Examples:
        You can invoke this function as a member of ``RegressionResults``. See below.
        This is the preferred way to call this function.

        >>> import btplus
        >>> import statsmodels.api as sm
        >>> from bokeh.plotting import output_notebook
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> mod = sm.OLS(...)
        >>> res = mod.fit()
        >>> res.panel()

        You can also tabulate multiple results into a single figure as shown below.

        >>> import btplus
        >>> import statsmodels.api as sm
        >>> from bokeh.plotting import show, output_notebook
        >>> from bokeh.models import Tabs, Panel
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> m1 = sm.OLS(...)
        >>> m2 = sm.OLS(...)
        >>> r1 = m1.fit()
        >>> r2 = m2.fit()
        >>> f1 = r1.panel(display=False)
        >>> f2 = r2.panel(display=False)
        >>> p1 = Panel(title='First Result', child=f1)
        >>> p2 = Panel(title='Second Result', child=f2)
        >>> output = Tabs(tabs=[p1, p2])
        >>> show(output)
         """
    data = pd.concat([res.fittedvalues.rename('Fitted'), pd.Series(res.model.data.orig_endog, name='Observed'), res.resid.rename('Residual')], axis=1)
    plots = {}
    plots['Model Fit'] = time_series(data, xlabel='Date', ylabel='Value',height=300, width=800)
    plots['Probability Plot'] = prob_plot(data['Residual'].rename('Residual'))

    plots['Residual ACF'] = acf_plot(data['Residual'].rename('Residual'))

    if exog_oos is not None:
        pred = res.predict(sm.add_constant(exog_oos)).rename('Predicted')
        if endog_oos is not None:
            oos = pd.concat([pred, endog_oos.rename('Observed')], axis=1)
            oos['Error'] = oos['Observed'] - oos['Predicted']
        else:
            oos = pred

        plots['Prediction'] = time_series(oos, xlabel='Date', ylabel='Value', title='Out-of-Sample Test')

    figure = Tabs(tabs=[Panel(title=key, child=plots[key]) for key in plots])
        

    tables = res.summary().tables
    labels = ['Model', 'Parameters', 'Residuals']
    table_tabs = []
    for t, l in zip(tables, labels):
        table_tabs.append(Panel(title=l, child=Div(text=t.as_html(), width=500, height=150)))

    tabulated_table = Tabs(tabs=table_tabs)

    out = column([figure, tabulated_table])

    if display:
        show(out)
    return out

def pandas_to_bokeh(df, method='html', **kwargs):
    """
    Converts a DataFrame to a bokeh table. 
    
    Args:
        df (DataFrame): The DataFrame to convert.
        method (str): Must be either 'html' or 'datatable'. The difference is entirely aesthetic.
        
    Returns:
        A bokeh table element

    Examples:
        You can invoke this function as a member of ``DataFrame``. See below.
        This is the preferred way to call this function.

        >>> import btplus
        >>> import pandas as pd
        >>> from bokeh.plotting import output_notebook, show
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> df = pd.DataFrame(...)
        >>> table = df.to_bokeh_table()
        >>> show(table)

        You can also tabulate multiple tables into a single figure as shown below.

        >>> import btplus
        >>> from statsmodels.regression.rolling import RollingOLS
        >>> from bokeh.plotting import show, output_notebook
        >>> from bokeh.models import Tabs, Panel
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> df1 = pd.DataFrame(...)
        >>> df2 = pd.DataFrame(...)
        >>> t1 = df1.to_bokeh_table()
        >>> t2 = df2.to_bokeh_table()
        >>> p1 = Panel(title='First Table', child=t1)
        >>> p2 = Panel(title='Second Table', child=t2)
        >>> output = Tabs(tabs=[p1, p2])
        >>> show(output)
    """
    if method=='html':
        data = Div(text=df.to_html(), **kwargs)
    elif method=='datatable':
        src = ColumnDataSource(df)
        cols = [TableColumn(field=x, title=x.title()) for x in src.data.keys()]
        data = DataTable(source=src, columns=cols)
    else:
        raise ValueError(":arg: method must be 'html' or 'datatable', not '{}'".format(method))

    return data

def interactive_candlestick(prices, title=None, tooltips=True, date_slider=True, height=400, width=1000):

    """
    Returns an interactive candlestick chart with volume bars and date range slider. Note that ``prices`` requires 
    the following columns: adj_open, adj_close, adj_high, adj_low, adj_volume. 

    This function was designed to work with daily prices from QuoteMedia (see ``btplus.data.QuoteMedia``). Your mileage may
    vary with data from other sources.
    
    Args:
        prices (DataFrame): A DataFrame containing adjusted OHLCV values plus dates.
        title (str, optional): A title for the plot.
        tooltips (bool): Enables hover tooltips.
        date_slider (bool): Enables date slider (does not work in streamlit)
        height (int): Figure height.
        width (int): Figure width.
        
    Returns:
        An interactive candlestick plot
    """

    df = prices.copy().set_index('date').asfreq('d').fillna(0).reset_index()
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    df = prices.copy()
    df['volume_millions'] = (df.adj_volume/1000000).round(3)
    df['direction'] = 'unknown'
    df['date_str'] = df.date.dt.strftime('%b %d, %y')
    inc = df.close > df.open
    dec = df.open > df.close
    df.loc[inc, 'direction'] = 'increasing'
    df.loc[dec, 'direction'] = 'decreasing'
    src = ColumnDataSource(df)
    incr_view = CDSView(source=src, filters=[GroupFilter(column_name='direction', group='increasing')])
    decr_view = CDSView(source=src, filters=[GroupFilter(column_name='direction', group='decreasing')])

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p1 = figure(x_axis_type="datetime", tools=TOOLS, plot_width=width, plot_height=height, title=title)
    p1.xaxis.visible = False
    p1.xaxis.major_label_orientation = 3.1415926535/4
    p1.grid.grid_line_alpha=0.3

    w = 12*60*60*1000 # half day in ms
    p1.segment(x0='date', y0='adj_high', x1='date', y1='adj_low', source=src, color="black")
    p1.vbar(x='date', width=w, top='adj_open', bottom='adj_close', fill_color="#D5E1DD", line_color="black",source=src, view=incr_view)
    p1.vbar(x='date', width=w, top='adj_open', bottom='adj_close', fill_color="#F2583E", line_color="black", source=src, view=decr_view)
    p1.xaxis.axis_label = 'Date'
    p1.yaxis.axis_label = 'Price ($)'

    if tooltips:
        p1_ttips = [('Open', '@adj_open{($ 0.00 a)}'), ('High', '@adj_high{($ 0.00 a)}'), ('Low', '@adj_low{($ 0.00 a)}'), ('Close', '@adj_close{($ 0.00 a)}'), ('Volume', '@{volume_millions}M'), ('Date', '@date_str')]
        htool = HoverTool(tooltips=p1_ttips, mode='vline')
        p1.add_tools(htool)

    p2 = figure(x_axis_type="datetime", tools="", toolbar_location=None, plot_width=width, plot_height=int(0.75*height), x_range=p1.x_range)
    p2.xaxis.major_label_orientation = 3.1415926535/4
    p2.grid.grid_line_alpha=0.3
    p2.vbar(df.date, w, df.volume_millions, [0]*df.shape[0])
    p2.xaxis.axis_label = 'Date'
    p2.yaxis.axis_label = 'Volume (Millions of Shares)'

    plots = [p1, p2]

    if date_slider:
        date_range_slider = DateRangeSlider(title="Date Range", start=df.date.values[0], end=df.date.values[-1], \
                                                value=(df.date.values[0], df.date.values[-1]), step=10, width=int(0.8*width))
        date_range_slider.js_link('value', p1.x_range, 'start', attr_selector=0)
        date_range_slider.js_link('value', p1.x_range, 'end', attr_selector=1)
        plots.insert(1, date_range_slider)

    return column(plots)

def rolling_regression_panel(res, display=True, **fig_kwargs):
    """
    Displays rolling regression results as a dynamic panel.
    
    Args:
        res (RollingRegressionResults): The regression results to visualize.
        display (bool): If True, displays the panel. Otherwise, the panel is returned. The latter
        is useful for tabulating multiple results into the same graphic object.

    Returns:
        A dynamic bokeh layout

    Examples:
        You can invoke this function as a member of ``RollingRegressionResults``. See below.
        This is the preferred way to call this function.

        >>> import btplus
        >>> from statsmodels.regression.rolling import RollingOLS
        >>> from bokeh.plotting import output_notebook
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> mod = RollingOLS(...)
        >>> res = mod.fit()
        >>> res.panel()

        You can also tabulate multiple results into a single figure as shown below.

        >>> import btplus
        >>> from statsmodels.regression.rolling import RollingOLS
        >>> from bokeh.plotting import show, output_notebook
        >>> from bokeh.models import Tabs, Panel
        >>> output_notebook() #include this if running in Jupyter Notebook
        >>> m1 = RollingOLS(...)
        >>> m2 = RollingOLS(...)
        >>> r1 = m1.fit()
        >>> r2 = m2.fit()
        >>> f1 = r1.panel(display=False)
        >>> f2 = r2.panel(display=False)
        >>> p1 = Panel(title='First Result', child=f1)
        >>> p2 = Panel(title='Second Result', child=f2)
        >>> output = Tabs(tabs=[p1, p2])
        >>> show(output)
         """
    data = {}
    data['Parameters'] = res.params
    data['Parameter Std. Error'] = res.bse

    pvals = pd.DataFrame(res.pvalues, index=res.params.index, columns=res.params.columns)
    pvals['F Statistic'] = res.f_pvalue
    data['P-Values'] = pvals

    data['Mean Square Error'] = pd.DataFrame({'Model': res.mse_model, 'Residual': res.mse_resid, 'Total': res.mse_total})
    pred = (res.params.shift()*sm.add_constant(res.model.data.orig_exog)).sum(axis=1).rename('Prediction')
    obs = res.model.data.orig_endog.rename('Observed')
    cond = pred!=0
    prediction_df = pd.concat([pred.loc[cond], obs.loc[cond]], axis=1)
    prediction_df['Error'] = prediction_df['Observed'] - prediction_df['Prediction']
    data['Prediction'] = prediction_df

    tabs = []
    for k in data:
        tabs.append(
            Panel(title=k, child=time_series(data[k], xlabel='Date', title='{}'.format(k), **fig_kwargs))
        )

    out = Tabs(tabs=tabs)

    if display:
        show(out)
    return out

def pandas_hiplot(df):
    """
    Generates a HiPlot object from a dataframe. See https://github.com/facebookresearch/hiplot 
    for more information. 
     
    Args:
        df (DataFrame): The table to generate a figure from. 
        
    Returns:
        A HiPlot.

    Example:
        This function can also be called as the member function of a DataFrame instance as shown below.
        Note that btplus must be imported.

        >>> import btplus
        >>> import pandas as pd
        >>> mydf = pd.DataFrame(...)
        >>> mydf.hiplot()

    """
    return hip.Experiment.from_dataframe(df).display()

def extend_objects():
    pd.DataFrame.to_bokeh_table = pandas_to_bokeh
    pd.DataFrame.bokeh_time_series = time_series
    pd.DataFrame.bokeh_hist = hist
    pd.DataFrame.hiplot = pandas_hiplot
    RegressionResults.panel = regression_panel
    RollingRegressionResults.panel = rolling_regression_panel









