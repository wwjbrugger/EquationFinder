from abc import ABC, abstractmethod
import numpy as np 

class AbstractOperator(ABC):

    @abstractmethod
    def __init__(self, node):
        self.num_child = None   # number of child nodes for the operator
        self.node = node
        self.invertible = None  # is the operation invertible
        self.valid_min_value = -np.inf
        self.valid_max_value = np.inf

    @abstractmethod
    def prefix_notation(self, call_node_id, kwargs):
        """
        Specifies the node-level behavior for printing
         the syntax tree in prefix order.
        :param call_node_id: int
        :return: str
        """
        pass

    @abstractmethod
    def infix_notation(self, call_node_id, kwargs):
        """
        Specifies the node-level behavior for printing
         the syntax tree in infix order.
        :param call_node_id: int
        :param call_node_id: dict with node information
        :return: str
        """
        pass

    @abstractmethod
    def evaluate_subtree(self, call_node_id, dataset, kwargs):
        """
         Specifies the node-level behavior for evaluating
         the syntax tree.
        :param call_node_id:
        :param dataset:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def delete(self):
        """
        Specifies the node-level behavior for deleting
         the syntax tree.
        :return:
        """
        
    def operator_data_range(self, variable):
        """
        Specifies the node-level behavior for deleting
         the syntax tree.
        :return:
        """
        min_value = self.valid_min_value
        max_value = self.valid_max_value
        depends_on_variable = False
        for i in range(self.num_child):
            c_min_value, c_max_value, depends_on_variable =\
                self.node.list_children[i].math_class.operator_data_range(variable)
            if depends_on_variable:
                if c_min_value > min_value:
                    min_value = c_min_value
                if c_max_value < max_value:
                    max_value = c_max_value
        return min_value, max_value, depends_on_variable
            

    
