bt_view - A Data Application for Viewing Backtests
====================================

bt_view is currently in alpha stage - if you find a bug, please submit an issue.

What is bt_view?
-----------

**bt_view** is an implementation of `bt <https://github.com/pmorissette/bt>`_ in a web application using `Streamlit <https://streamlit.io/>`_. It includes
a user interface for creating, reading, and visualizing backtests. 

Installing and running bt_view
-------------

To install bt_view, clone the repo, the install the package from the setup.py file:

.. code-block:: bash

    $ git clone https://github.com/Quantitative-Investment-Society/bt_view.git
    $ cd bt_view
    $ pip install .

To run the web application, use streamlit to run the main file.

.. code-block:: bash

    $ cd bt_view
    $ streamlit run main.py

Since bt_view has many dependencies, it's strongly recommended to install the `Anaconda Scientific Python
Distribution <https://store.continuum.io/cshop/anaconda/>`_, especially on Windows. This distribution 
comes with many of the required packages pre-installed, including pip. Once Anaconda is installed, the above 
command should complete the installation. 


