import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# df = pd.read_csv('jou_1995_loading.csv')
# index = [df['MEA_weight_fraction'], df['temperature']]
# df = pd.DataFrame(df.to_numpy()[:, 2:], index, columns=list(df.columns[2:]))
#
# for i1, w_MEA in enumerate(df.index.unique('MEA_weight_fraction')):
#
#     for i2, T in enumerate(df.index.unique('temperature')):
#         x = []
#         y_data = []
#         y_model = []
#         for i, row in df.loc[w_MEA, T].iterrows():
#             print(i, row['CO2_loading'], row['CO2_pressure'], row['total_pressure'])
#         print(i2)


    # for w_MEA in df["MEA_weight_fraction"].unique():
    #
    #     for k, row_2 in df.loc[df["MEA_weight_fraction"] == w_MEA].iterrows():
    #
    #         fig = plt.figure(k)
    #         ax = fig.subplots()
    #
    #         for j, T in enumerate(df["temperature"].unique()):
    #             x = []
    #             y_data = []
    #             y_model = []
    #             for i, row in df.loc[df["temperature"] == T].iterrows():
    #                 n_data += 1
    #                 CO2_loading = row["CO2_loading"]
    #                 x.append(CO2_loading)
    #                 y_data.append(row["CO2_pressure"])
    #                 candidate_idx = np.where(df["CO2_pressure"].values == row["CO2_pressure"])
    #
    #                 assert len(candidate_idx) == 1
    #                 assert len(candidate_idx[0]) == 1
    #
    #                 fug_CO2 = pyo.value(blk[candidate_idx[0][0]].fug_phase_comp["Liq", "CO2"]) / 1e3
    #                 y_model.append(fug_CO2)
    #
    #             for y_data_pt, y_model_pt in zip(y_data, y_model):
    #                 P_CO2_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)
    #
    #             ax.semilogy(x, y_data, linestyle="none", marker="o", color=colors[j])
    #             ax.semilogy(x, y_model, linestyle="-", color=colors[j], label=f"T = {T}")
    #         ax.set_xlabel("CO2 Loading")
    #         ax.set_ylabel("CO2 vapor pressure (kPa)")
    #         ax.set_title(
    #             f"Dataset: {blk.name[:-8]}, MEA mass {w_MEA:.2%}: {len(estimated_vars_scaled)} params fitted - Rel Error: {P_CO2_rel_err / n_data:.4f} - Obj: {pyo.value(m.obj):.4f} - H: {1 / W[0]:.4f}",
    #             wrap=True)
    #         ax.legend()
    #         fig.show()