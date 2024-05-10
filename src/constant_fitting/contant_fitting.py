from src.syntax_tree.syntax_tree import SyntaxTree
import pandas as pd


def refit_all_constants(finished_state, args):
    complete_syntax_tree, initial_dataset = reconstruct_complete_syntax_tree(finished_state=finished_state, args=args)
    if complete_syntax_tree.num_constants_in_complete_tree > 0:
        complete_syntax_tree.constants_in_tree['num_fitted_constants']= 0
        complete_syntax_tree.fit_constants(
            call_node_id=0,
            dataset=initial_dataset,
            mode='evaluate')
    return complete_syntax_tree, initial_dataset



def  reconstruct_complete_syntax_tree(finished_state, args):
    if finished_state.previous_state is None:
        # State has no history, so we assume the dataset saved with the dataset is the initial one
        initial_dataset = finished_state.observation['data_frame']
        return finished_state.syntax_tree, initial_dataset
    else:
        actions, initial_dataset = get_all_actions_and_initial_dataset(
            state=finished_state,
            actions=[]
        )
        syntax_tree = SyntaxTree(grammar=finished_state.syntax_tree.grammar,
                                 args=finished_state.syntax_tree.args
                                 )
        i = 0
        while len(syntax_tree.nodes_to_expand) > 0:
            action =  actions[i]
            syntax_tree.expand_node_with_action(
                node_id=syntax_tree.nodes_to_expand[0],
                action=action,
                build_syntax_tree_token_based=args.build_syntax_tree_token_based
            )
            i += 1
        syntax_tree.constants_in_tree = finished_state.syntax_tree.constants_in_tree
        return syntax_tree, initial_dataset


def get_all_actions_and_initial_dataset(state, actions):
    if check_for_first_state(state):
        return actions, state.observation['data_frame']
    else:
        actions.insert(0, state.production_action)
        actions, initial_dataset = get_all_actions_and_initial_dataset(
            state=state.previous_state,
            actions=actions
        )
    return actions, initial_dataset

def  check_for_first_state(state):
    if state.previous_state:
        return False
    else:
        return True
