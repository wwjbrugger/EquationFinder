from pathlib import Path

import pandas as pd

def save_panda_dataframes(save_folder, dict_measurements, approach, args):
    save_folder = Path(save_folder) / f'{approach}'
    save_folder.mkdir(exist_ok=True, parents=True)
    for key, values in dict_measurements.items():
        str_formula = f"{values['formula']}".replace('/', 'div')
        df = values['df']
        if args.store_multiple_versions_of_one_equation:
            df.to_csv(save_folder / f"{str_formula}__{key}.csv", index= False)
        else:
            df.to_csv(save_folder / f"{str_formula}.csv", index=False)




