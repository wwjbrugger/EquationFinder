import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
path = Path('/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/data_grammar_1/run_3/test/pandas')
# List of file names
file_names = [
    # 'Nguyen-1_test.csv', 'Nguyen-2_test.csv', 'Nguyen-3_test.csv',
    # 'Nguyen-4_test.csv',
    # 'Nguyen-5_test.csv', 'Nguyen-6_test.csv',
    # 'Nguyen-7_test.csv', 'Nguyen-8_test.csv',
    'Nguyen-9_test.csv',
    'Nguyen-10_test.csv', 'Nguyen-11_test.csv', 'Nguyen-12_test.csv'
]

# Read and concatenate data from all CSV files
dfs = [pd.read_csv(path / file) for file in file_names]
#data = pd.concat(dfs, ignore_index=True)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, data in enumerate(dfs):
# Scatter plot
    ax.scatter(data['x_0'], data['x_1'], data['y'],
               label=file_names[i], alpha=1)

ax.legend()

# Labels
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('y')

plt.show()