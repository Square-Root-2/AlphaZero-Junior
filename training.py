from AlphaZeroConfig import AlphaZeroConfig
from Game import Game
from MoveGenerator import MoveGenerator
import MoveIndexing
from Network import Network
from Node import Node
from ReplayBuffer import ReplayBuffer

import keras
import math
import multiprocessing
from multiprocessing import Lock, Pool, Value
from multiprocessing.managers import SyncManager
import numpy as np
import tensorflow as tf
from typing import List

game_number = 1
move_number = 1

# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.


def alphazero(config: AlphaZeroConfig):
    SyncManager.register("ReplayBuffer", ReplayBuffer, exposed=("save_game", "sample_pseudobatch", "is_empty"))

    manager = SyncManager()
    manager.start()

    replay_buffer = manager.ReplayBuffer(config)
    step = manager.Value('i', 255)
    screen_lock = manager.Lock()

    pool = Pool(config.num_actors)
    for i in range(config.num_actors):
        pool.apply_async(func=run_selfplay, args=(config, replay_buffer, step, screen_lock))

    train_network(config, replay_buffer, step, screen_lock)

    pool.terminate()

    return latest_network(config.training_steps)


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, replay_buffer: ReplayBuffer, step: Value, screen_lock: Lock):
    global game_number, move_number
    while True:
        game_step = step.value
        network = latest_network(game_step)
        game = play_game(config, network, screen_lock, game_step)
        replay_buffer.save_game(game)
        game_number += 1
        move_number = 1


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network, screen_lock: Lock, step: int):
    global game_number, move_number
    game = Game()
    root = Node(0)
    while not game.terminal() and len(game.history) - 1 < config.max_moves:
        action = run_mcts(config, game, network, root)
        game.apply(action)
        game.store_search_statistics(root)
        root = root.children[action]
        with screen_lock:
            print(multiprocessing.current_process().name + ", Step " + str(step) + ", Game " + str(game_number) + ", Move " + str(move_number))
            game.history[-1].print()
        move_number += 1
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network, root: Node):
    if not root.expanded():
        evaluate(root, game, network)
    add_exploration_noise(config, root)

    simulations = sum(root.children[action].visit_count for action in root.children.keys())

    while simulations < config.num_simulations:
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        if scratch_game.terminal():
            value = scratch_game.terminal_value(game.to_play())
        else:
            value = evaluate(node, scratch_game, network)

        backpropagate(search_path, value, scratch_game.to_play())

        simulations += 1

    return select_action(config, game, root)


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if len(game.history) - 1 < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts)
    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
    value, policy_logits = network.inference(tf.expand_dims(game.make_image(-1), axis=0))
    value = int(value)
    policy_logits = tf.squeeze(policy_logits, axis=[0])

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################


##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, replay_buffer: ReplayBuffer, step: Value, screen_lock: Lock):
    while replay_buffer.is_empty():
        continue

    network = Network()
    optimizer = tf.keras.optimizers.SGD(config.learning_rate_schedule,
                                        config.momentum)
    for i in range(step.value, config.training_steps):
        loss = 0
        gradient = [tf.Variable(np.zeros(weights.shape), dtype=tf.float32) for weights in network.get_weights()]
        for j in range(config.batch_size // config.pseudobatch_size):
            screen_lock.acquire()
            print("Training Step " + str(i + 1) + ", Pseudobatch " + str(j + 1))
            print()
            screen_lock.release()
            pseudobatch = replay_buffer.sample_pseudobatch()
            loss += process_pseudobatch(network, pseudobatch, gradient)
        with tf.GradientTape() as tape:
            for weights in network.get_weights():
                loss += config.weight_decay * tf.nn.l2_loss(weights)
        d_gradient = tape.gradient(loss, network.get_weights(), unconnected_gradients=tf.UnconnectedGradients.ZERO)
        for k in range(len(gradient)):
            gradient[k].assign_add(d_gradient[k])
        optimizer.apply_gradients(zip(gradient, network.get_weights()))
        save_network(i + 1, network)
        step.value += 1
        if (i + 1) % config.checkpoint_interval == 0:
            save_network(i + 1, network, game_generator=False)
        screen_lock.acquire()
        print("Loss: " + str(float(loss)))
        print()
        screen_lock.release()


def process_pseudobatch(network: Network, pseudobatch: List[tf.Tensor], gradient: List[tf.Variable]):
    loss = 0
    with tf.GradientTape() as tape:
        value, policy_logits = network.inference(pseudobatch[0])
        loss += tf.reduce_sum(
            tf.losses.mean_squared_error(value, pseudobatch[1]) +
            tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=pseudobatch[2]))

    d_gradient = tape.gradient(loss, network.get_weights(), unconnected_gradients=tf.UnconnectedGradients.ZERO)
    for k in range(len(gradient)):
        gradient[k].assign_add(d_gradient[k])

    return loss

######### End Training ###########
##################################


def softmax_sample(visit_counts):
    visit_counts_sum = sum(visit_count for (visit_count, action) in visit_counts)
    k = np.random.choice(
        len(visit_counts),
        p=[visit_count / visit_counts_sum for (visit_count, action) in visit_counts])
    return visit_counts[k]


class UniformNetwork(Network):
    def __init__(self):
        Network.__init__(self, is_uniform=True)

    def inference(self, image):
        return tf.constant([0]), tf.constant([MoveIndexing.TOTAL_MOVES * [1 / MoveIndexing.TOTAL_MOVES]])  # Value, Policy


def latest_network(step: int) -> Network:
    if step > 0:
        model = keras.models.load_model('model')
        network = Network()
        network.model = model
        return network
    else:
        return UniformNetwork()  # policy -> uniform, value -> 0


def save_network(step: int, network: Network, game_generator=True):
    if game_generator:
        network.model.save('model')
    else:
        network.model.save('model' + str(step))


if __name__ == '__main__':
    MoveGenerator.initialize()
    config = AlphaZeroConfig()
    config.num_actors = 6
    config.checkpoint_interval = 100
    alphazero(config)
