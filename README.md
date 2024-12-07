# ExemPy

ExemPy is a library of functions for implementing the Generalized Context Model, and routines for simulating speech perception experiment. This repository also serves as a workspace for my 2024 dissertation in Linguistics at UC Berkeley and follow-up work.

ExemPy is built around the [pandas library](https://pandas.pydata.org/) and the [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) structure. This repository makes extensive use of [jupyter notebooks](https://jupyter.org/). To get started using ExemPy, I recommend setting up [JupyterLab](https://jupyter.org/install). 

From there, you can begin by following along with the notebooks in my "Dissertation demos" folder, starting with "Basics," with your own data. You'll need a dataset (for example, a csv file) that has multiple observations to function as your "exemplar cloud." Each observation should have category labels assigned to the observation (e.g., "vowel" or "speaker"), and features with values (e.g., formant frequencies).

## Install using pip
To install, do: 
`pip install git+https://github.com/emilyremirez/ExemPy`

