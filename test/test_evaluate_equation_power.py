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
                'y': [1, 8, 27, 64]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** 3 x'.split())


        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([1, 8, 27, 64], list(result))

    def test_power_subtree_1(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [1, 8, 27, 64]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())


        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([1, 8, 27, 64], list(result))



    def test_power_prefix_notion_1(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())
        result = syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=1)

        self.assertEqual('/ log y log - x 0'.split(),  result[1].split() )

    def test_power_prefix_notion_2(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())
        result = syntax_tree.rearrange_equation_prefix_notation(new_start_node_id=260)

        self.assertEqual('** / 1 + 1 2 y '.split(),  result[1].split() )

    def test_power_infix_notion_0(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=-1)

        self.assertEqual('( x - 0 ) ** ( 1 +  2 ) '.split(),  result[1].split() )

    def test_power_infix_notion_1(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=1)

        self.assertEqual('log ( y ) /  log ( ( x -  0 ) ) '.split(),  result[1].split() )

    def test_power_infix_notion_2(self):
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='** + 1 2  - x 0'.split())
        result = syntax_tree.rearrange_equation_infix_notation(new_start_node_id=260)

        self.assertEqual('  y  ** ( 1 / ( 1 + 2 ) )'.split(),  result[1].split() )

