from src.utils.error import NonFiniteError
import numpy as np
class HelperObjectConstantFitting():
    def __init__(self, node_to_evaluate, node_id):
        self.node_to_evaluate = node_to_evaluate
        self.node_id = node_id


    def evaluate_subtree(self, xdata, *params):
        syntax_tree = self.node_to_evaluate.tree
        for i in range(syntax_tree.constants_in_tree['num_fitted_constants'],
                       syntax_tree.num_constants_in_complete_tree):
            parameter_index = i - syntax_tree.constants_in_tree['num_fitted_constants']
            syntax_tree.constants_in_tree[f"c_{i}"]['value']= params[parameter_index]
        temp = self.node_to_evaluate.math_class.evaluate_subtree(
            call_node_id=self.node_id,
            dataset=xdata,
            kwargs=syntax_tree.constants_in_tree
        )
        if np.all(np.isfinite(temp)):
            return temp
        else:
            raise NonFiniteError


