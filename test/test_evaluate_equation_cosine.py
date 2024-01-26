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
    #     syntax_tree.prefix_to_syntax_tree(prefix='+ cos x  1'.split())
    #     syntax_tree.print()

    def test_cosine_evaluate_sub_tree(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.54030231, 0.58385316, 0.0100075,  0.34635638]

                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ cos x  1'.split())


        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([1.54, 0.58, 0.01, 0.35], list(result.round(2)))

    def test_cosine_evaluate_sub_tree_1(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.54030231, 0.58385316, 0.0100075,  0.34635638]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ cos x  1'.split())


        result = syntax_tree.evaluate_subtree(node_id=1, dataset=df)
        np.testing.assert_almost_equal([0.54, - 0.42, -0.99, -0.65], list(result.round(2)))

    def test_cosine_evaluate_sub_tree_5(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1.54030231, 0.58385316, 0.0100075,  0.34635638]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ cos x  1'.split())
        result = syntax_tree.evaluate_subtree(node_id=16, dataset=df)
        np.testing.assert_almost_equal([1,1,1,1], list(result))



    
