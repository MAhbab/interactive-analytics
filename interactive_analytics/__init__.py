from .visual import extend_objects

import webbrowser
import os

def open_docs():
    f = os.path.join(
        'file://', os.path.dirname(__file__), '../docs/_build/singlehtml/index.html'
    )
    webbrowser.open(f, new=2)

extend_objects()
