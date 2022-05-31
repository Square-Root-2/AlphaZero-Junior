from AlphaZeroConfig import AlphaZeroConfig
from MoveGenerator import MoveGenerator

import training


def train(num_actors=6, checkpoint_interval=10):
    MoveGenerator.initialize()
    config = AlphaZeroConfig()
    config.num_actors = num_actors
    config.checkpoint_interval = checkpoint_interval
    training.alphazero(config)


if __name__ == '__main__':
    train()
