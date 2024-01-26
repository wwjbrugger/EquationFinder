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
    #     syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())
    #     syntax_tree.print()

    def test_power_subtree_0(self):
        dict = {'x': [1, 2, 3, 4],
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='exp x'.split())

        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([2.72, 7.39, 20.09, 54.60], list(result), decimal=2)

    def test_power_subtree_1(self):
        dict = {'x': [1, 2, 3, 4],
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- exp - x 0 2'.split())

        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([0.72, 5.39, 18.09, 52.60], list(result), decimal=2)

        # ________________________________________________________



    def test_power_prefix_notion_1(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- exp - x 0 2'.split())
        result = syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=1)

        self.assertEqual('+ y 2'.split(), result[1].split())

    def test_power_prefix_notion_2(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- exp - x 0 2'.split())
        result = syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=10)

        self.assertEqual('- x ln + y 2'.split(), result[1].split())

    # ________________________________________________________

    def test_power_infix_notion_0(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- exp - x 0 2'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=1)

        self.assertEqual(' ( y + 2  )'.split(), result[1].split())

    def test_power_infix_notion_1(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- exp - x 0 2'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=10)

        self.assertEqual(' ( x - ln ( y + 2 ) )'.split(), result[1].split())




