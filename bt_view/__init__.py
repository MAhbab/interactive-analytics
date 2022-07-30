from .data import Eod
from .visualization import ResultVisual

import webbrowser
import os

def open_docs():
    f = os.path.join(
        'file://', os.path.dirname(__file__), '../docs/singlehtml/index.html'
    )
    webbrowser.open(f, new=2)