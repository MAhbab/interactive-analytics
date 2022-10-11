btplus - Backtesting Functionality Add-ons
==========================================
btplus is currently in alpha stage - if you find a bug, please submit an issue.

What is btplus?
---------------
**btplus** is an add-on to `bt <https://github.com/pmorissette/bt>`_. It includes two big additions: data source APIs and 
visualizations.

Installing and running btplus
-----------------------------
To install btplus, clone the repo, the install the package from the setup.py file:

.. code-block:: bash

    $ git clone https://github.com/Quantitative-Investment-Society/btplus.git
    $ cd btplus
    $ pip install .

Documentation
-------------
Documentation can only be viewed locally from within a script as seen below. 

.. code-block:: python

    import btplus
    btplus.open_docs()

Features
--------
Below are just a few of the features included in btplus!

* **Simple and Fast Data Retrieval**
    Pull stock data from QuoteMedia, economic data from FRED, and risk factor data from Eugene Fama's website
    with a single function call (each). Data is stored in a *persistent* cache and easily updated.

    .. code-block:: python

        #quickly retrieve OHLCV data for stocks and ETFs
        from btplus.data import QuoteMedia
        data = QuoteMedia.get(['SPY', 'AMZN'])
        amzn = data['AMZN']

* **Interactive Panels**
    Generate modular, interactive visuals for commonly used objects in `bt <https://github.com/pmorissette/bt>`_, 
    `pandas <https://pandas.pydata.org/>`_, and `statsmodels <https://www.statsmodels.org/stable/index.html>`_, with 
    more on the way. Here are just a few:

    * **Dynamic Candlestick Chart**
        View price data in Jupter Notebook cleanly and interactively.

        .. image:: https://imgur.com/mpXO0Ew.gif

    * **Regression Results**

        .. image:: https://imgur.com/NQqZtVL.gif

    * **Backtest Results HiPlot** (Using `HiPlot <https://github.com/facebookresearch/hiplot>`_)

        .. image:: https://imgur.com/dQOYC4W.png


Since btplus has many dependencies, it's strongly recommended to install the `Anaconda Scientific Python
Distribution <https://store.continuum.io/cshop/anaconda/>`_, especially on Windows. This distribution 
comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above 
command should complete the installation. 


