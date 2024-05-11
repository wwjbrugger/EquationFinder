import tensorflow as tf
from pathlib import Path
import pandas as pd
from definitions import ROOT_DIR
import numpy as np
from src.utils.get_grammar import get_grammar_from_string
from src.generate_datasets.grammars import get_grammars
from src.generate_datasets.dataset_generator import get_all_symbols_usable
class EquationPreprocessDummy():
    """
    Class to read data dynamically to transformer model
    """

    def __init__(self, args, train_test_or_val, grammar):
        self.args = args
        self.grammar = grammar
        self.train_test_or_val = train_test_or_val
        self.symbol_hash_dic = self.get_hash_values_for_symbols()
        self.symbol_lookup = self.cast_dic_to_lookup_table(self.symbol_hash_dic)

    def get_hash_values_for_symbols(self):
        symbol_hash_dic = {}
        all_symbols = [str(symbol) for symbol in get_all_symbols_usable(self.grammar)]
        all_symbols.sort()
        for i, symbol in enumerate(all_symbols):
            symbol_hash_dic[symbol.strip()] = np.float32(i + 1)
        self.vocab_size = len(all_symbols) + 1

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
                                        f"pandas")

        dataset = tf.data.experimental.make_csv_dataset(
            column_names=self.column_names,
            file_pattern=file_pattern,
            batch_size=1,
            column_defaults=self.type_list,
            header=True
        )

        preprocessed_dataset = self.preprocess(dataset)
        iterator = iter(preprocessed_dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE))
        return iterator

    def get_num_production_rules(self):
        return len(self.grammar._productions)

    def preprocess(self, dataset):
        RuntimeWarning('This method should be overwritten by a child class.'
                       'If you want to use the dummy object ignore this warning ')
        return dataset


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

def  get_dict_token_to_action(grammar):
    token_to_action = {}
    for i, production in enumerate(grammar._productions):
        token_to_action[production._rhs[0]] = i
    return token_to_action

def equation_to_action_sequence(equation, token_to_action,
                                equation_to_action_sequence,
                                grammar):
    if not equation in equation_to_action_sequence:
        action_sequence = []
        for token in equation.split():
            action_sequence.append(token_to_action[token])
        action_sequence.append(grammar.terminal_action)
        equation_to_action_sequence[equation] = action_sequence
    return equation_to_action_sequence[equation]



