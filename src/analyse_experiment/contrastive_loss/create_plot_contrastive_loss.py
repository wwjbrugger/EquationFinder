from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

path = Path('/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/src/analyse_experiment/contrastive_loss/contrative_loss_11_03.csv')

# Read and concatenate data from all CSV files
dfs = pd.read_csv(path)
# data = pd.concat(dfs, ignore_index=True)
experiment_name = {
    'Bi_LSTM_Measurement_Encoder abs_max_y__MAX': 'iteration',
    'Bi_LSTM_Measurement_Encoder abs_max_y__MIN': 'Bi-LSTM',
    'DatasetTransformer abs_max_y__MIN': 'Dataset Transformer',
    'DatasetTransformer abs_max_y__lin_transform__MIN': 'lin. Dataset Transformer',
    'MLP_Measurement_Encoder abs_max_y__lin_transform__MIN': 'lin. MLP',
    'MeasurementEncoderPicture abs_max_y__lin_transform__MIN': 'lin. Image',
    'TextTransformer abs_max_y__lin_transform__MIN': 'lin. Text Transformer'
}

dfs_short = dfs.loc[:, list(experiment_name.keys())]
dfs_short = dfs_short.rename(columns=experiment_name)
fig = plt.figure(figsize=(9, 6))
plt.rcParams.update({
    # 'text.usetex': True,
    'font.family': 'Helvetica',
    'axes.labelsize': 18,
    'text.latex.preamble': r'usepackage{amsmath}',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 16,
}),
ax = fig.add_subplot(111)
for experiment in experiment_name.values():
    if experiment != 'iteration':
        ax.errorbar(x=dfs_short.loc[:, 'iteration'], y=dfs_short.loc[:, experiment],
                    label=experiment)
ax.set_xlabel('Iteration')
ax.set_ylabel('Contrastive Loss')
ax.legend()
plt.show()
# ['Step', 'Bi_LSTM_Measurement_Encoder abs_max_y',
#    'Bi_LSTM_Measurement_Encoder abs_max_y__MIN',
#    'Bi_LSTM_Measurement_Encoder abs_max_y__MAX',
#    'Bi_LSTM_Measurement_Encoder abs_max_y__lin_transform',
#    'Bi_LSTM_Measurement_Encoder abs_max_y__lin_transform__MIN',
#    'Bi_LSTM_Measurement_Encoder abs_max_y__lin_transform__MAX',
#    'DatasetTransformer abs_max_y', 'DatasetTransformer abs_max_y__MIN',
#    'DatasetTransformer abs_max_y__MAX',
#    'DatasetTransformer abs_max_y__lin_transform',
#    'DatasetTransformer abs_max_y__lin_transform__MIN',
#    'DatasetTransformer abs_max_y__lin_transform__MAX',
#    'MLP_Measurement_Encoder abs_max_y__lin_transform',
#    'MLP_Measurement_Encoder abs_max_y__lin_transform__MIN',
#    'MLP_Measurement_Encoder abs_max_y__lin_transform__MAX',
#    'TextTransformer abs_max_y__lin_transform',
#    'TextTransformer abs_max_y__lin_transform__MIN',
#    'TextTransformer abs_max_y__lin_transform__MAX',
#    'MeasurementEncoderPicture abs_max_y__lin_transform',
#    'MeasurementEncoderPicture abs_max_y__lin_transform__MIN',
#    'MeasurementEncoderPicture abs_max_y__lin_transform__MAX'],)
# Plotting

# for i, data in enumerate(dfs):
#     # Scatter plot
#     ax.scatter(data['x_0'], data['x_1'], data['y'],
#                label=file_names[i], alpha=1)
#
#
#
# # Labels
# ax.set_xlabel('x_0')
# ax.set_ylabel('x_1')
# ax.set_zlabel('y')
#
# plt.show()
