import camelot
import ctypes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from ctypes.util import find_library
# print(find_library("".join(("gsdll", str(ctypes.sizeof(ctypes.c_voidp) * 8), ".dll"))))
#
# tables = camelot.read_pdf('Xu.pdf', pages='3', flavor='stream')
# for i, table in enumerate(tables):
#     filename = r'C:\Users\Tanner\Documents\git\MEA\data\Xu_' + str(i) + '.csv'
#     table.df.to_csv(filename)

df = pd.read_csv(r"C:\Users\Tanner\Documents\git\MEA\data\data_sets_to_load\Xu_2011_VLE.csv")

df['temperature'] = np.round(df['temperature'].to_numpy(), -1)
df = df.sort_values(['temperature', 'CO2_loading'])
df.to_csv(r"C:\Users\Tanner\Documents\git\MEA\data\data_sets_to_load\Xu_2011_sorted_VLE.csv")
