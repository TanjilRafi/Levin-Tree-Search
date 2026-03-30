import numpy as np

class UniformModel:
    def get_probabilities(self, _):
        return np.ones(4) / 4


class Model:
    def __init__(self, use_reversed=False):
        self._table = {}
        self._learning_rate = 0.2
        self._use_reversed = use_reversed

    def _ensure_context(self, context):
        if context not in self._table:
            self._table[context] = np.zeros(4, dtype=float)
        return self._table[context]

    def _softmax(self, logits):
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits)

    def get_probabilities(self, context):
        weights = self._ensure_context(context)
        return self._softmax(weights)

    def _gradient_update(self, context, action):
        weights = self._ensure_context(context)
        probs = self._softmax(weights)
        gradient = probs.copy()
        gradient[action] -= 1.0
        self._table[context] = weights - self._learning_rate * gradient

    def update(self, path):
        states = path.get_states()
        actions = path.get_actions()

        for state, action in zip(states, actions):
            regular_ctx, reversed_ctx = state.get_reversed_context()

            # version 1
            self._gradient_update(regular_ctx, action)

            # version 2
            if self._use_reversed and reversed_ctx != regular_ctx:
                self._gradient_update(reversed_ctx, action)