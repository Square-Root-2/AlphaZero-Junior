from AlphaZeroConfig import AlphaZeroConfig
from MoveGenerator import MoveGenerator
import training


def train():
    MoveGenerator.initialize()
    config = AlphaZeroConfig()
    net = training.alphazero(config)
    net.model.save('model')


train()
