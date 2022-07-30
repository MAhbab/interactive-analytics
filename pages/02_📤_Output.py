from bt_view import ResultVisual
import bt_view.visualization.utils as vu
import streamlit as st

def metric_options():
    return {
        'Returns': vu.Returns,
        'Drawdown': vu.Drawdown,
        'Sharpe': vu.Sharpe,
        'Sortino': vu.Sortino,
        'Daily': vu.Daily,
        'Monthly': vu.Monthly,
        'Yearly': vu.Yearly
    }

def run():

    res = st.session_state['result']

    if res is None:
        st.error('ERROR: You must read in results to view them.')
        st.stop()

    rv = ResultVisual(res)

    if st.sidebar.checkbox('Time Series Plots'):
        st.subheader('Time Series Plots')
        bk_key = st.selectbox('Backtest', res.backtests)
        fig = rv.paginated_plot(bk_key)
        st.bokeh_chart(fig)

    if st.sidebar.checkbox('Stats - Table'):
        st.subheader('Stats - Table')
        stats = rv.stats_table()
        st.bokeh_chart(stats)

    if st.sidebar.checkbox('HiPlot'):
        st.subheader('HiPlot')
        hiplot_container = st.container()

        with hiplot_container:
            opts = metric_options()
            key = st.selectbox('Metric options', opts)
            selection = opts[key]
            monthly_metrics = selection._member_map_.values()
            rv.stats_hiplot(*monthly_metrics)

if __name__=='__main__':
    run()