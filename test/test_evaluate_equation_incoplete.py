import unittest

from src.syntax_tree.syntax_tree import SyntaxTree
import pandas as pd
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
    #     dict = {'x': [1, 2, 3, 4],
    #             'y': [4,  1.75 ,  1.3333333333333333, 1.1875]
    #             }
    #     df = pd.DataFrame(dict)
    #     syntax_tree = SyntaxTree(grammar=None, args=self.args)
    #     syntax_tree.prefix_to_syntax_tree(prefix='- + cos 0 / 3 * x A sin 0'.split())
    #     print(syntax_tree.rearrange_equation_infix_notation(new_start_node_id=-1))
    #     syntax_tree.print()

    def test_evaluate_subtree(self):
        dict = {'x': [1, 2, 3, 4],
                'y': [4,  1.75 ,  1.3333333333333333, 1.1875]
                }
        df = pd.DataFrame(dict)
        syntax_tree = SyntaxTree(grammar=None, args=self.args)
        syntax_tree.prefix_to_syntax_tree(prefix='- + cos 0 / 3 * x x sin 0'.split())
        result = syntax_tree.evaluate_subtree(node_id=0, dataset=df)
        np.testing.assert_almost_equal([4,  1.75 ,  1.3333333333333333, 1.1875], list(result))




