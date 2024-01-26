import unittest

from src.syntax_tree.syntax_tree import SyntaxTree
import pandas as pd
import time
from pcfg import PCFG
from itertools import product


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        grammar_string ="""  
        S -> '+' S S [0.1]| '-' S S [0.1]| '*' S S [0.1]| 'sin' Inner_Function [0.1] 
        S -> 'cos' Inner_Function [0.1] | 'log' Inner_Function [0.1]  
        S -> 'x_0' [0.05] | 'x_1' [0.05]
        S -> '**' Potenz Variable [0.3]
        S -> '1' [0.0]
        S -> '0.5' [0.0]
        S -> '2' [0.0]
        Potenz -> '6' [0.0]| '5' [0.1] | '4' [0.1] | '3' [0.1] |'2' [0.1] | '0.5' [0.1] | 'x_1' [0.5]
        Inner_Function -> '**' Potenz Variable [0.4] | 'x_0' [0.2] | 'x_1' [0] | '+' SUM SUM [0.4]  
        SUM -> '**' Potenz Variable [0.5] | '1' [0.2] | 'x_0' [0.15] | 'x_1' [0.15]
        Variable -> 'x_0' [0.5] | 'x_1' [0.5]
           """

        self.grammar = PCFG.fromstring(grammar_string)

        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree = 10
        self.args.max_constants_in_tree = 5
        self.args.number_equations = 10
        self.args.num_calls_sampling = 10
        self.args.sample_with_noise = False
        self.args.how_to_select_node_to_delete = 'random'


    def test_production_for_tree(self):
        equation_str = '+  ** 3 x_0   ** 2 x_0'
        syntax_tree_true = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree_true.prefix_to_syntax_tree(prefix=equation_str.split())
        possible_production = syntax_tree_true.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_0(self):
        syntax_tree_true = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree_true.prefix_to_syntax_tree(prefix='+ + ** 3 x_0 ** 2 x_0  x_0 '.split())
        possible_production = syntax_tree_true.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_1(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ + + ** 4 x_0 ** 3 x_0 ** 2 x_0  x_0 '.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_2(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ + + + ** 5 x_0 ** 4 x_0 ** 3 x_0 ** 2 x_0  x_0 '.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_3(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ + + + + ** 6 x_0 ** 5 x_0 ** 4 x_0 ** 3 x_0 ** 2 x_0  x_0 '.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_5(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- * sin ** 2 x_0 cos x_0 1 '.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))


    def test_nguyen_6(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ sin x_0 sin + x_0 ** 2 x_0 '.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_7(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ log + x_0 1 log + ** 2 x_0 1 '.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_8(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** 0.5 x_0'.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_9(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ sin x_0 sin ** 2 x_1'.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_10(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='* 2 * sin x_0 cos x_1'.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_11(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** x_1 x_0'.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))

    def test_nguyen_12(self):
        syntax_tree = SyntaxTree(grammar=self.grammar, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- + - ** 4 x_0 ** 3 x_0  * 0.5 ** 2 x_1 x_1'.split())
        possible_production = syntax_tree.possible_production_for_tree()
        self.assertTrue(len(possible_production))
