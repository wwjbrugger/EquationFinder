def save_used_symbols_to_file(grammar_tree, save_folder):
    save_folder.mkdir(exist_ok=True, parents=True)
    with open(save_folder / 'symbols.txt', "a") as file:
        file.writelines(str(symbol) + ', ' for symbol in grammar_tree.all_usable_symbols)
        file.writelines('[, ], ')


def save_production_rules_to_file(grammar_tree, save_folder):
    save_folder.mkdir(exist_ok=True, parents=True)
    with open(save_folder / 'production_rules.txt' , "a") as file:
     file.writelines(str(production) + '\n' for production in grammar_tree.grammar._productions)

def save_grammar_to_file(grammar_string, save_folder):
    save_folder.mkdir(exist_ok=True, parents=True)
    with open(save_folder / 'grammar.txt', "a") as file:
        file.writelines(line for line in grammar_string)
