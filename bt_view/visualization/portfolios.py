#%%
import bt
import pandas as pd
import streamlit as st

from bt.algos import Algo
from typing import Hashable, List

#TODO: Implement more selection algos in SelectConfig

class RunConfig:

    def __init__(
        self,
        container=None,
        run_freq=True,
        run_after_date=False,
        run_after_days=False,
        run_every_n_periods=False,
        run_if_out_of_bounds=False,
        use_or_algo=False,
        key=None
        
    ) -> None:
        self.container = container or st.container()
        self._active_options = []

        if run_freq:
            self._active_options.append(self.frequency)
        
        if run_after_days:
            self._active_options.append(self.after_days)

        if run_after_date:
            self._active_options.append(self.after_date)

        if run_every_n_periods:
            self._active_options.append(self.every_n_periods)

        if run_if_out_of_bounds:
            self._active_options.append(self.if_out_out_of_bounds)

        self.use_or = use_or_algo
        self.key = key or ''

    def frequency(self) -> Algo:
        options = [
            'RunDaily',
            'RunWeekly',
            'RunMonthly',
            'RunQuarterly',
            'RunYearly'
        ]

        help = 'How often your algo will run'
        selection = st.selectbox('Frequency', options, format_func=lambda x: x.replace('Run', ''), help=help, key='run_freq_{}'.format(self.key))
        algo = getattr(bt.algos, selection)
        return algo()

    def after_days(self) -> Algo:
        selection = st.number_input(
            'Start Delay (Days)',
            1,
            format='%i',
            key='run_after_days_{}'.format(self.key),
            help='Number of trading days to wait before running algo.'
        )
        
        return bt.algos.RunAfterDays(selection)

    def after_date(self):
        pass

    def every_n_periods(self):
        pass

    def if_out_out_of_bounds(self):
        pass


    def configure(self) -> List[Algo]:
        cols = self.container.columns(len(self._active_options))
        algos = []

        for c, fn in zip(cols, self._active_options):
            with c:
                a = fn()
                algos.append(a)

        if self.use_or:
            return [bt.algos.Or(algos)]
        
        return algos

class SelectConfig:

    def __init__(self, container=None, key=None) -> None:
        self.container = container or st.container()
        self.key = key or ''

    def configure(self):
        with self.container:
            options = [bt.algos.SelectAll, bt.algos.SelectRandomly]
            algo = st.selectbox(
                'Selection Algo',
                options,
                format_func=lambda x: x.__name__.replace('Select', ''),
                key='select_algo_{}'.format(self.key),
                help='Choose a method for selecting assets.'
            )

        return [algo()]

class WeightConfig:

    def __init__(self, container=None, key=None, limit_deltas=False, limit_weights=False, scale_weights=False) -> None:
        self.container = container or st.container()
        self.key = key or ''
        self._scale_weights_flag = scale_weights
        self._limit_deltas_flag = limit_deltas
        self._limit_weights_flag = limit_weights

    def _set_lookback(self) -> pd.DateOffset:
        ndays = st.number_input(
            'Lookback (Days)',
            1,
            value=90,
            step=1,
            format='%i',
            key='weighting_lookback_{}'.format(self.key),
            help='Set a lookback window for calculating security weights.'
        )

        return pd.DateOffset(days=ndays)

    def _set_covariance_estimation_method(self):
        pass

    def configure(self) -> List[Algo]:
        options = [bt.algos.WeighEqually, bt.algos.WeighRandomly, bt.algos.WeighTarget]
        key = 'weight_algo_selection_{}'.format(self.key)
        algo = st.selectbox(
            'Weight Algo',
            options,
            format_func=lambda x: x.__name__.replace('Weigh', ''),
            key=key,
            help='Select a weight scheme'
        )

        if algo is bt.algos.WeighTarget:
            return [self.read_from_csv(key=':'.join(key, 'target_weight'))]

        else:
            return [algo()]

    def read_from_csv(self, key: Hashable) -> bt.Algo:
        f = st.file_uploader('Upload Weights', 'csv', False, key=key)
        if f is None:
            return bt.algos.WeighEqually()

        df = pd.read_csv(f)
        if not isinstance(df.index, pd.DatetimeIndex):
            st.error("ERROR: Uploaded file doesn't seem to be a time-index table. Application halted")
            st.stop()

        return bt.algos.WeighTarget(df)

def configure() -> List[Algo]:

    run_algos = RunConfig().configure()
    select_algos = SelectConfig().configure()
    weigh_algos = WeightConfig().configure()
    algos = run_algos + select_algos + weigh_algos
    algos.append(bt.algos.Rebalance())
    return algos

            






        

