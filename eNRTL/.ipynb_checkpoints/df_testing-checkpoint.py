import pandas as pd
import numpy as np
from tabulate import tabulate
from IPython.display import display, HTML

columns = ['Parameter Name', 'Value', 'Uncertainty', 'Percent']
parameters = {}
parameters['Parameter Name'] = r'$C_{H_2O},{(MEAH^+, MEACOO^-)}$'
parameters['Value'] = 1.00
parameters['Uncertainty'] = 0.001
parameters['Percent'] = 1

# pd.set_option("display.latex.repr", True)
df = pd.DataFrame(parameters, index=[1], columns=columns)
display(HTML(df.to_html()))
