import numpy as np

class UniformModel:
    def get_probabilities(self, _):
        return np.ones(4) / 4

class Model:
    def __init__(self):
        self._table = {}
        self._learning_rate = 0.2

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
    
    def update(self, path):
        pass