import unittest

from src.syntax_tree.syntax_tree import SyntaxTree
import pandas as pd
import time
import numpy as np
class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        class Namespace():
            def __init__(self):
                pass

        self.args = Namespace()
        self.args.logging_level = 40
        self.args.max_branching_factor = 2
        self.args.max_depth_of_tree= 5

    # def test_print(self):
    #     syntax_tree = SyntaxTree(grammar=None, args=self.args)
    #     syntax_tree.prefix_to_syntax_tree(prefix='+ x / x 4'.split())
    #     syntax_tree.print()
    def test_addition(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.25, 2.5, 3.75,  5]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x / x 4'.split())
        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([1.25, 2.5, 3.75,  5], list(result))

    def test_addition_subtree_1(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.25, 2.5, 3.75,  5]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x / x 4'.split())


        result = syntax_tree.evaluate_subtree(node_id=1, dataset=df)
        np.testing.assert_almost_equal([1,2,3,4], list(result))

    def test_addition_subtree_5(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.25, 2.5, 3.75,  5]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x / x 4'.split())
        result = syntax_tree.evaluate_subtree(node_id=16, dataset=df)
        np.testing.assert_almost_equal([0.25, 0.5, 0.75, 1], list(result))

    def test_addition_subtree_8(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.25, 2.5, 3.75,  5]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ x / x 4'.split())
        result = syntax_tree.evaluate_subtree(node_id=24, dataset=df)
        np.testing.assert_almost_equal([4,4,4,4], list(result))

