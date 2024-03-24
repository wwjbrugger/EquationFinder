"""
Main function of the codebase. This file is intended to call different parts of our pipeline based on console arguments.

To add new games to the pipeline, add your string_query-class constructor to the 'game_from_name' function.
https://github.com/kaesve/muzero
"""
import random
from datetime import datetime
from src.config_ngedn import Config
from src.coach import Coach
from src.neural_nets.rule_predictor_skeleton import RulePredictorSkeleton
from src.game.find_equation_game import FindEquationGame
from src.mcts.classic_mcts import ClassicMCTS
from src.mcts.amex_mcts import AmEx_MCTS
import tensorflow as tf
import numpy as np
import wandb
from definitions import ROOT_DIR
from src.utils.get_grammar import read_grammar_file
from src.utils.copy_weights import copy_dataset_encoder_weights_from_pretrained_agent
def run():
    args = Config.arguments_parser()
    args.ROOT_DIR = ROOT_DIR
    time_string = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    unique_dir = f"{time_string}_{args.seed}"
    wandb_path = ROOT_DIR / '.wandb' /  args.experiment_name / \
                 f"{unique_dir}"
    wandb_path.mkdir(parents=True, exist_ok=True)
    wandb.init(entity="wwjbrugger", config=args.__dict__,
               project="neural_guided_symbolic_regression",
               sync_tensorboard=True, tensorboard=True,
               dir=wandb_path, mode=args.wandb,
               name=args.experiment_name)
    wandb.log({'Job_ID': args.job_id})

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Set up tensorflow backend.
    if int(args.gpu) >= 0:
        device = tf.DeviceSpec(device_type='GPU', device_index=int(args.gpu))
    else:
        device = tf.DeviceSpec(device_type='CPU', device_index=0)

    grammar = read_grammar_file(args=args)

    # preprocessor = get_preprocessor_class(args=args)
    # reader_train = preprocessor(args=args, train_test_or_val='train')
    # iter_train = reader_train.get_datasets()
    game = FindEquationGame(
        grammar,
        args,
        train_test_or_val='train'
    )

    game_test = FindEquationGame(
        grammar,
        args,
        train_test_or_val='test'
    )

    learnA0(g=game,
            args=args,
            run_name=args.experiment_name,
            game_test=game_test
            )
    wandb.log(
        {
            f"sucessful": True
        }
    )


def learnA0(g, args, run_name: str, game_test) -> None:
    """
    Train an AlphaZero agent on the given environment with the specified configuration. If specified within the
    configuration file, the function will load in a previous model along with previously generated data.
    :param game_test: 
    :param args:
    :param g: Game Instance of a Game class that implements environment logic. Train agent on this environment.
    :param run_name: str Run name to store data by and annotate results.
    """
    print("Testing:", ", ".join(run_name.split("_")))

    # Extract neural network and algorithm arguments separately
    rule_predictor_train = RulePredictorSkeleton(
        args=args,
        reader_train=g.reader
    )
    rule_predictor_test = RulePredictorSkeleton(
        args=args,
        reader_train=game_test.reader
    )
    checkpoint_train, manager_train = load_pretrained_net(
        args = args,
        rule_predictor=rule_predictor_train,
        game = g
    )
    checkpoint_test, _ = load_pretrained_net(
        args=args,
        rule_predictor=rule_predictor_test,
        game=g
    )
    if args.MCTS_engine == 'Endgame':
        search_engine = AmEx_MCTS
    elif args.MCTS_engine == 'Normal':
        search_engine = ClassicMCTS
    else:
        raise AssertionError(f"Engine: {args.MCTS_engine} not defined!")

    c = Coach(
        game=g,
        game_test=game_test,
        rule_predictor=rule_predictor_train,
        rule_predictor_test=rule_predictor_test,
        args=args,
        search_engine=search_engine,
        run_name=run_name,
        checkpoint_train=checkpoint_train,
        checkpoint_manager=manager_train,
        checkpoint_test=checkpoint_test
    )

    c.learn()


def load_pretrained_net(args, rule_predictor, game):
    dataset_number = args.data_path.name
    experiment_name = f"{args.experiment_name}/{args.seed}"
    net = rule_predictor.net
    checkpoint_path_current_model = ROOT_DIR / 'saved_models' / dataset_number / \
                                 experiment_name
    print(f"Model will be saved at {checkpoint_path_current_model}")

    checkpoint_current_model = tf.train.Checkpoint(
        step=tf.Variable(1),
        net=net
    )
    manager_train = tf.train.CheckpointManager(
        max_to_keep = 30,
        step_counter=checkpoint_current_model.step,
        checkpoint=checkpoint_current_model,
        directory=str(checkpoint_path_current_model / 'tf_ckpts'),
        checkpoint_interval=10
    )
    initialize_net(args, checkpoint_current_model, game)

    if len(args.path_to_complete_model) > 0:
        restore_path = ROOT_DIR / args.path_to_complete_model
        if restore_path.suffix != '':
            raise RuntimeError(f"Your path to the complete model has an suffix: {restore_path.suffix} \n "
                                 f"the restore operation wants to have the path in the form *path_to_checkpoint/tf_chpts/ckpt-x* \n"
                                 f" Most likely you add the path to the index file \n"
                                 f"Your path is: {restore_path}" )

        checkpoint_current_model.restore(f"{restore_path}")
        print("Restored from {}".format(f"{ROOT_DIR / args.path_to_complete_model }"))

    elif manager_train.latest_checkpoint:
        checkpoint_current_model.restore(manager_train.latest_checkpoint).assert_consumed()

        print("Restored from {}".format(manager_train.latest_checkpoint))
    else:
        checkpoint_current_model.restore(manager_train.latest_checkpoint)
        print("Initializing from scratch.")
    print_num_weights_of_model(net)

    copy_dataset_encoder_weights_from_pretrained_agent(
        args=args,
        checkpoint_current_model=checkpoint_current_model,
        game=game
    )

    return checkpoint_current_model, manager_train


def print_num_weights_of_model(net):
    num_actor = np.sum([np.prod(v.get_shape().as_list()) for v in net.actor.trainable_variables])
    num_critic = np.sum([np.prod(v.get_shape().as_list()) for v in net.critic.trainable_variables])
    num_encoder_measurement = np.sum([np.prod(v.get_shape().as_list()) for v in net.encoder_measurement.trainable_variables])
    num_encoder_tree = np.sum([np.prod(v.get_shape().as_list()) for v in net.encoder_tree.trainable_variables])
    num_total = num_actor + num_critic + num_encoder_measurement + num_encoder_tree
    print(f"Parameter: \n "
          f"   measurement: {num_encoder_measurement} \n"
          f"   syntax tree: {num_encoder_tree} \n"
          f"   actor: {num_actor} \n"
          f"   critic: {num_critic} \n"
          f"   total: {num_total} \n"
          )


def initialize_net(args, checkpoint_current_model, game):
    iter = game.reader.get_datasets()
    net = checkpoint_current_model.net
    data_dict = next(iter)
    prepared_syntax_tree = np.zeros(
        shape=(1, args.max_tokens_equation),
        dtype=np.float32)
    prepared_dataset = net.encoder_measurement.prepare_data(
        data=data_dict
    )
    net(
        input_encoder_tree=prepared_syntax_tree,
        input_encoder_measurement=prepared_dataset
    )


def get_run_name(config_name: str, architecture: str, game_name: str) -> str:
    """ Macro function to wrap various ModelConfig properties into a run name. """
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{config_name}_{architecture}_{game_name}_{time}"


if __name__ == "__main__":
    run()
    # cProfile.run('run()', filename='profile.prof')
