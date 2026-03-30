
import copy
import heapq
import math
import numpy as np

class TreeNode:
    def __init__(self, parent, game_state, p, d, action):
        self._game_state = game_state
        self._p = p
        self._g = d
        self._action = action
        self._parent = parent
        self._probabilitiy_distribution_a = None
    
    def __eq__(self, other):
        """
        Verify if two tree nodes are identical by verifying the 
        game state in the nodes. 
        """
        return self._game_state == other._game_state
    
    def __lt__(self, other):
        """
        Function less-than used by the heap
        """
        return self._levin_cost < other._levin_cost    
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
    def set_probability_distribution_actions(self, d):
        self._probabilitiy_distribution_a = d
        
    def get_probability_distribution_actions(self):
        return self._probabilitiy_distribution_a
    
    def set_levin_cost(self, c):
        self._levin_cost = c
    
    def get_p(self):
        """
        Returns the pi cost of a node
        """
        return self._p
    
    def get_depth(self):
        """
        Returns the depth of a node
        """
        return self._g
    
    def get_game_state(self):
        """
        Returns the game state represented by the node
        """
        return self._game_state
    
    def get_parent(self):
        """
        Returns the parent of the node
        """
        return self._parent
    
    def get_action(self):
        """
        Returns the action taken to reach node stored in the node
        """
        return self._action

class Trajectory():
    def __init__(self, states, actions):
        self._states = states
        self._actions = actions
        
    def get_states(self):
        return self._states
    
    def get_actions(self):
        return self._actions
    
    def length(self):
        return len(self._states)
    
class BFSLevin():
            
    def recover_path(self, tree_node):
        states = []
        actions = []
        
        state = tree_node.get_parent()
        action = tree_node.get_action()
        
        while not state.get_parent() is None:
            states.append(state.get_game_state())
            actions.append(action)
            
            action = state.get_action()
            state = state.get_parent()
            
        states.append(state.get_game_state())
        actions.append(action)
        
        return Trajectory(states, actions)        
     

    def get_levin_cost(self, node):
        """
        Levin cost in log-space
        """
        d = node.get_depth()
        if d <= 0:
            return float("-inf")
        return np.log(d) - node.get_p()

    def search(self, initial_state, model, budget=-1):
        """
        Returns solution_cost, expansions, trajectory and
        IF NOT solved within budget -1, expansions, None
        """
        open_list = []
        closed = set()
        expansions = 0

        # Root node
        root = TreeNode(None, copy.deepcopy(initial_state), 0.0, 0, -1)

        # Store action probabilities at root
        p_root = model.get_probabilities(root.get_game_state().get_context())
        p_root_log = np.log(np.maximum(p_root, 1e-300))
        root.set_probability_distribution_actions(p_root_log)
        root.set_levin_cost(self.get_levin_cost(root))

        heapq.heappush(open_list, root)

        while open_list:
            if budget != -1 and expansions >= budget:
                return -1, expansions, None

            parent = heapq.heappop(open_list)

            if parent in closed:
                continue
            closed.add(parent)

            expansions += 1

            if parent.get_game_state().is_solution():
                return parent.get_depth(), expansions, self.recover_path(parent)

            actions = parent.get_game_state().successors_parent_pruning(parent.get_action())
            parent_log_probs = parent.get_probability_distribution_actions()

            for a in actions:
                child_state = copy.deepcopy(parent.get_game_state())
                child_state.apply_action(a)

                child_log_pi = parent.get_p() + parent_log_probs[a]
                child_depth = parent.get_depth() + 1

                child_node = TreeNode(parent, child_state, child_log_pi, child_depth, a)

                child_probs = model.get_probabilities(child_state.get_context())
                child_log_probs = np.log(np.maximum(child_probs, 1e-300))
                child_node.set_probability_distribution_actions(child_log_probs)

                child_node.set_levin_cost(self.get_levin_cost(child_node))
                heapq.heappush(open_list, child_node)

        return -1, expansions, None
    
