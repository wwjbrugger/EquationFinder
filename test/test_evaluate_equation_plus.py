import unittest

from src.syntax_tree.syntax_tree import SyntaxTree
import pandas as pd
import time

class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree= 6
    def test_addition(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [6, 8, 10, 12]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x + x 4'.split())

        [syntax_tree.evaluate_subtree(node_id=0,dataset=df) for i in range(1)]

        for i in range(1):
            _, equation = syntax_tree.rearrange_equation_infix_notation(-1)
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                y = eval(equation, row_dict, globals())
        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        self.assertEqual([6,8,10,12], list(result))

    def test_addition_subtree_1(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [6, 8, 10, 12]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x + x 4'.split())
        result = syntax_tree.evaluate_subtree(node_id=32, dataset=df)
        self.assertEqual([5,6,7,8], list(result))

    def test_addition_subtree_2(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [6, 8, 10, 12]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x + x 4'.split())

        result = syntax_tree.evaluate_subtree(node_id=48, dataset=df)
        self.assertEqual([4,4,4,4], list(result))

