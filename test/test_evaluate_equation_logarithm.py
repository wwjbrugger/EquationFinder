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
        self.args.max_branching_factor = 6
        self.args.max_depth_of_tree= 5

    # def test_print(self):
    #     syntax_tree = SyntaxTree(grammar=None, args=self.args)
    #     syntax_tree.prefix_to_syntax_tree(prefix='+ 1 log x'.split())
    #     syntax_tree.print()

    def test_logarithm_subtree_0(self):
        dict = {'x': [10, 100, 1000],
                'y': [2, 3, 4 ]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ 1 log x'.split())


        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([ 2, 3, 4 ], list(result))

    def test_logarithm_subtree_1(self):
        dict = {'x': [10, 100, 1000],
                'y': [2, 3, 4]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ 1 log x'.split())
        result = syntax_tree.evaluate_subtree(node_id=260, dataset=df)
        np.testing.assert_almost_equal([1, 2, 3], list(result))

  
#________________________________________________________


    def test_logarithm_infix_notion_0(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ 1 log x'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=-1)

        self.assertEqual('( 1 + log ( x ) )'.split(),  result[1].split() )

    def test_logarithm_infix_notion_1(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='+ 1 log x'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=261)

        self.assertEqual(' 10 ** ( y - 1 ) '.split(), result[1].split())