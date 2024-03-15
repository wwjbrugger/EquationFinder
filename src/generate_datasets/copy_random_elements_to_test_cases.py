import os
import random
import shutil

# Path to your folder
folder_path = '/data_grammar_1/run_1/test/pandas'

# # List all files in the folder
# files = os.listdir(folder_path)
# random.shuffle(files)
# selected_files = files[:15]

selected_files = [
    'x_0__2225.csv',
    ' ln ( x_1 )__5962.csv',
    '  x_0  **  6  __1092.csv',
    '(   x_0  **  4   +  ( (   x_0  **  4   +  ( ( x_1 * ( x_0 * x_0 )  )  -   x_0  **  2   )  )  *  sin ( ( c_0 +  c_1 )  ) )  ) c_0_3.62_c_1_2.52___5532.csv',
    '( x_1 +  ( (   x_0  **  3   *  ln (   x_0  **  2   ) )  * (  ln ( c_0 ) * (   x_0  **  5   +    x_0  **  4   )  )  )  ) c_0_2.64___4294.csv',
    '(   x_0  **  x_1   +  (   x_0  **  2   * x_0 )  ) __1875.csv',
    '( x_1 +    x_0  **  3   ) __1509.csv',
    '(  ln (   x_0  **  2   ) +    x_0  **  3   ) __3187.csv',
    '(  ln ( ( c_0 +  x_1 )  ) +  (  sin ( ( c_1 +    x_0  **  2   )  ) *  sin ( (   x_0  **  2   +  c_2 )  ) )  ) c_0_1.33_c_1_4.73_c_2_4.69___4358.csv',
    '( (   x_0  **  5   * ( ( x_1 +   ln ( ( c_0 *   x_0  **  2   )  ) )  *   x_0  **  2   )  )  +  x_1 ) c_0_1.63___9016.csv'
]
# Print the selected files
for i, file_name in enumerate(selected_files):
    folder_symbols = '/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/data_grammar_1/run_1'
    new_folder=f'/home/jbrugger/PycharmProjects/NeuralGuidedEquationDiscovery/data_grammar_1/self_{i}'

    original_path_production_rules = os.path.join(folder_symbols, f"production_rules.txt")
    original_path_symbols = os.path.join(folder_symbols, f"symbols.txt")
    original_path = os.path.join(folder_symbols, f"test/pandas/{file_name}")

    new_path_production_rules = os.path.join(new_folder, f"production_rules.txt")
    new_path_symbols = os.path.join(new_folder, f"symbols.txt")
    new_path_test = os.path.join(new_folder, f"test/pandas/{file_name}")
    new_path_train = os.path.join(new_folder, f"train/pandas/{file_name}")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        os.makedirs(os.path.join(new_folder, 'test/pandas'))
        os.makedirs(os.path.join(new_folder, 'train/pandas'))
    shutil.copy(original_path, new_path_test)
    shutil.copy(original_path, new_path_train)
    shutil.copy(original_path_symbols, new_path_symbols)
    shutil.copy(original_path_production_rules, new_path_production_rules)
    pass

