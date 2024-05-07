from src.syntax_tree.syntax_tree import SyntaxTree
import random
from pcfg import PCFG
from nltk.grammar import Nonterminal


class EquationGenerator():
    def __init__(self, grammar, args):
        self.grammar = grammar
        self.args = args
        self.all_usable_symbols = self.get_all_symbols_usable()
        self.dic_production_to_index = self.create_dic_production_to_index()


    def get_all_symbols_usable(self):
        terminal_symbols = set([str(key) for key in self.grammar._lexical_index.keys()])
        non_terminal_symbols = set([str(key) for key in  self.grammar._lhs_index.keys()])
        all_symbols = terminal_symbols.union(non_terminal_symbols)
        return list(all_symbols)

    def create_dic_production_to_index(self):
        """
        Dict which map rulers to a number
        :return:
        """
        dic_production_to_index = {}
        for i, production in enumerate(self.grammar._productions):
            dic_production_to_index[production] = i
        return dic_production_to_index

    def create_new_equation(self):
        action_sequence =  []
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        while not ( syntax_tree.complete or  syntax_tree.max_depth_reached or
        syntax_tree.max_constants_reached or syntax_tree.max_nodes_reached):
            node_id_to_expand = syntax_tree.nodes_to_expand[0]
            _ , production_index = self.sample_production_rule(
                syntax_tree=syntax_tree,
                node_id_to_expand = node_id_to_expand
            )
            try:
                syntax_tree.expand_node_with_action(
                    node_id=node_id_to_expand,
                    action=production_index,
                    build_syntax_tree_eager = True
                )
                action_sequence.append(production_index)
            except:
                print(f"Error with action : {production_index}")
        return syntax_tree, action_sequence

    def sample_production_rule(self, syntax_tree, node_id_to_expand):

        node_symbol = syntax_tree.dict_of_nodes[node_id_to_expand].node_symbol
        possible_productions = self.grammar._lhs_index[Nonterminal(node_symbol)]
        probabilities = [production.prob() for production in possible_productions]
        selected_production = random.choices(
            possible_productions,
            weights=probabilities)[0]
        production_index = self.dic_production_to_index[selected_production]
        return selected_production, production_index


