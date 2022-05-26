from AlphaZeroConfig import AlphaZeroConfig
from Game import Game
from Network import Network
from Node import Node
from ReplayBuffer import ReplayBuffer
from SharedStorage import SharedStorage

import math
import numpy
import tensorflow as tf
import threading
from typing import List
from threading import Semaphore

index = {}
game_number = {}
move_number = {}
screenlock = Semaphore()

# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.


def alphazero(config: AlphaZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for i in range(config.num_actors):
        thread = threading.Thread(target=run_selfplay, name=str(i), args=(config, storage, replay_buffer))
        index[thread.name] = i
        game_number[thread.name] = 1
        move_number[thread.name] = 1
        thread.start()

    while len(replay_buffer.buffer) == 0:
        continue

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
        game_number[threading.current_thread().name] += 1


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
    game = Game()
    root = Node(0)
    while not game.terminal() and len(game.history) - 1 < config.max_moves:
        action = run_mcts(config, game, network, root)
        game.apply(action)
        game.store_search_statistics(root)
        root = root.children[action]
        move_number[threading.current_thread().name] += 1
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

    screenlock.acquire()
    print("Thread " + str(index[threading.current_thread().name]) + ": " + str(
        config.num_simulations - simulations) + " simulations left (Game "
          + str(game_number[threading.current_thread().name]) + ", Move " + str(
        move_number[threading.current_thread().name]) + ")")
    screenlock.release()

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

        if (config.num_simulations - simulations) % 50 == 0:
            screenlock.acquire()
            print("Thread " + str(index[threading.current_thread().name]) + ": " + str(config.num_simulations - simulations) + " simulations left (Game "
                  + str(game_number[threading.current_thread().name]) + ", Move " + str(move_number[threading.current_thread().name]) + ")")
            screenlock.release()

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
    value, policy_logits = network.inference(game.make_image(-1))

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
    noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################


##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network()
    optimizer = tf.keras.optimizers.SGD(config.learning_rate_schedule,
                                        config.momentum)
    network.model.save('model0')
    for i in range(config.training_steps):
        screenlock.acquire()
        print()
        print("Training step: " + str(i))
        print()
        screenlock.release()
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)
        network.model.save('model' + str(i + 1))
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.keras.optimizers.SGD, network: Network, batch,
                   weight_decay: float):
    loss = 0
    with tf.GradientTape() as tape:
        for image, (target_value, target_policy) in batch:
            value, policy_logits = network.inference(image)
            loss += (
                    tf.cast(tf.keras.metrics.mean_squared_error([value], [target_value]), dtype=tf.float32) +
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy))

        for weights in network.get_weights():
            loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.apply_gradients(zip(tape.gradient(loss, network.model.trainable_weights), network.model.trainable_weights))


######### End Training ###########
##################################

def softmax_sample(visit_counts):
    visit_counts_sum = sum(visit_count for (visit_count, action) in visit_counts)
    k = numpy.random.choice(
            len(visit_counts),
            p=[visit_count / visit_counts_sum for (visit_count, action) in visit_counts])
    return visit_counts[k]
