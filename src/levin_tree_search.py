
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
        pass

    def search(self, initial_state, model, budget=-1):
        pass
 
