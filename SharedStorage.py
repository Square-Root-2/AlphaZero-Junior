from Network import Network


class SharedStorage(object):

    def __init__(self):
        self._networks = {0: Network()}

    def latest_network(self) -> Network:
        return self._networks[max(self._networks.keys())]

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
