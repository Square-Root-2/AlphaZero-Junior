class Game(object):

    def __init__(self, history=None):
        self.history = history or []
        self.child_visits = []
        self.num_actions = 4672

    # TODO
    def terminal(self):
        pass

    # TODO
    def terminal_value(self, to_play):
        pass

    # TODO
    def legal_actions(self):
        return []

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    # TODO
    def make_image(self, state_index: int):
        return []

    # TODO
    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2
