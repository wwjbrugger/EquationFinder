import tensorflow as tf
from pathlib import Path
import pandas as pd
from definitions import ROOT_DIR
import numpy as np
class EquationPreprocessDummy():
    """
    Class to read data dynamically to transformer model
    """

    def __init__(self, args, train_test_or_val):
        self.args = args
        self.train_test_or_val = train_test_or_val
        self.type_list, self.column_names = self.infer_record_defaults(approach=self.args.tree_representation)

        self.symbol_hash_dic = self.get_hash_values_for_symbols()
        self.symbol_lookup = self.cast_dic_to_lookup_table(self.symbol_hash_dic)

    def infer_record_defaults(self, approach='path_hash'):
        """
        Get the data type of the files which are read
        :param approach:
        :return:
        """
        info_frame = pd.read_csv(ROOT_DIR / self.args.data_path /
                                 f'{self.train_test_or_val}/info_{approach}.csv')
        dic = {"<class 'str'>": tf.string,
               "<class 'int'>": tf.int32,
               "<class 'float'>": tf.float32,
               "<class 'list'>": tf.string
               }
        data_type_array = info_frame.iloc[0].to_numpy()
        type_list = [dic[type_str] for type_str in data_type_array]

        column_name = info_frame.iloc[1].to_numpy()
        return type_list, column_name

    def get_hash_values_for_symbols(self):
        symbol_hash_dic = {}
        with open(ROOT_DIR / self.args.data_path / 'symbols.txt') as f:
            content = f.read().splitlines()
            symbols = content[0].split(sep=', ')
            for i, symbol in enumerate(symbols):
                symbol_hash_dic[symbol] = np.float32(i + 1)
        self.vocab_size = len(symbols) + 1

        return symbol_hash_dic

    def cast_dic_to_lookup_table(self, symbol_hash_dic):
        keys_tensor = tf.convert_to_tensor(list(symbol_hash_dic.keys()))
        vals_tensor = tf.convert_to_tensor(list(symbol_hash_dic.values()))
        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        symbol_lookup = tf.lookup.StaticHashTable(
            init,
            default_value=np.float32(-1000))
        return symbol_lookup

    def get_datasets(self):
        self.num_production_rules = self.get_num_production_rules()
        file_pattern = tf.io.gfile.glob(f"{ROOT_DIR / self.args.data_path}/{self.train_test_or_val}/"
                                        f"combined_data_set_{self.args.tree_representation}/*")

        dataset = tf.data.experimental.make_csv_dataset(
            column_names=self.column_names,
            file_pattern=file_pattern,
            batch_size=1,
            column_defaults=self.type_list,
            header=True
        )

        preprocessed_dataset = self.preprocess(dataset)
        iterator = iter(preprocessed_dataset.batch(self.args.batch_size_loading).prefetch(tf.data.experimental.AUTOTUNE))
        return iterator

    def get_num_production_rules(self):
        with open(ROOT_DIR / self.args.data_path / 'production_rules.txt') as f:
            content = f.read().splitlines()
        num_production_rules = len(content)
        return num_production_rules

    def preprocess(self, dataset):
        RuntimeWarning('This method should be overwritten by a child class.'
                       'If you want to use the dummy object ignore this warning ')
        return dataset

    def cast_string_symbol_to_integer(self, tensor):
        output_tensor = self.symbol_lookup.lookup(tensor)
        return output_tensor

    def map_tree_representation_to_int(self, symbol_list):
        current_tree_representation = self.symbol_lookup.lookup(
            tf.convert_to_tensor(symbol_list))
        current_tree_representation = self.pad_up_to(
            t=current_tree_representation,
            max_in_dims=[self.args.max_tokens_equation]
        )
        return current_tree_representation

    def pad_up_to(self, t, max_in_dims, constant_values=np.float32(0)):
        s = tf.shape(t)
        if s[0] < max_in_dims[0]:
            paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
            tensor = tf.pad(t, paddings, 'CONSTANT',
                            constant_values=constant_values)
        else:
            tensor = t[:max_in_dims[0]]
        return tf.reshape(tensor=tensor, shape=(1, max_in_dims[0]))



    def print_dictionary_dataset(self, dataset):
        """
        Helper function to print datasets
        :param dataset:
        :return:
        """
        for i, element in enumerate(dataset):
            print("Element {}:".format(i))
            for (feature_name, feature_value) in element.items():
                print('{:>14} = {}'.format(feature_name, feature_value))



