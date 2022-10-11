from . import data as sources
from .visual import extend_objects
from bt import Strategy, Backtest, Algo
import bt.core as core
import bt.algos as algos
import bt.backtest as backtest

import webbrowser
import os

def open_docs():
    f = os.path.join(
        'file://', os.path.dirname(__file__), '../docs/_build/singlehtml/index.html'
    )
    webbrowser.open(f, new=2)

extend_objects()