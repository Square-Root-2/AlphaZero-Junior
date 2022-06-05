from AlphaZeroConfig import AlphaZeroConfig
from Game import Game
from MoveGenerator import MoveGenerator
from Network import Network
from Node import Node
import training

import keras
import time


class AlphaZero(object):

    def __init__(self):
        MoveGenerator.initialize()
        self.config = AlphaZeroConfig()
        self.network = Network(keras.models.load_model('model'))

    def run_mcts(self, game: Game, root: Node, search_time):
        start = time.time()

        if not root.expanded():
            training.evaluate(root, game, self.network)

        training.add_exploration_noise(self.config, root)

        while time.time() - start < search_time:
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.expanded():
                action, node = training.select_child(self.config, node)
                scratch_game.apply(action)
                search_path.append(node)

            if scratch_game.terminal():
                value = scratch_game.terminal_value(game.to_play())
            else:
                value = training.evaluate(node, scratch_game, self.network)

            training.backpropagate(search_path, value, scratch_game.to_play())

        return self.select_action(root)

    @staticmethod
    def select_action(root: Node):
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]
        _, action = max(visit_counts)
        return action
