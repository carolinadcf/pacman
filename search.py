# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List
import heapq
from collections import deque


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    # # Source: https://stackoverflow.com/questions/77539424/breadth-first-search-bfs-in-python-for-path-traversed-and-shortest-path-taken
    # visited = {}  # {state: [parent, action,]}
    # queue = deque()  # queue of nodes to visit

    # start_state = problem.getStartState()

    # visited[start_state] = None  # start state has no parent
    # queue.append(start_state)

    # while queue:
    #     current_state = queue.popleft()

    #     # goal is reached, get path
    #     if problem.isGoalState(current_state):
    #         path = []
    #         while current_state:  # stop at start state
    #             previous_state = visited[current_state][
    #                 1
    #             ]  # get action needed to get to node from parent
    #             path.append(current_state)
    #             current_state = previous_state
    #         # path goes from goal to start, need to reverse it
    #         return list(reversed(path))

    #     for next_state, action, _ in problem.getSuccessors(current_state):
    #         if next_state not in visited:  # state has not been already
    #             queue

    return aStarSearch(problem)


def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    to_visit = []  # nodes to be explored (heap)
    # f(n) - total cost, g(n) - cost to get there, [steps,], state
    start_state = problem.getStartState()
    heapq.heappush(to_visit, (0 + heuristic(start_state, problem), 0, [], start_state))

    visited = {}  # nodes already visited {state: g(n)}

    while to_visit:
        f, g, path, current_state = heapq.heappop(
            to_visit
        )  # get cheapest node from heap

        if problem.isGoalState(current_state):
            return path

        # que el nodo no se haya visitado o si ya ha sido visitado que el (nuevo) coste sea menor al anterior coste registrado
        if current_state not in visited or g < visited[current_state]:
            visited[current_state] = g  # menor (nuevo) coste en llegar al nodo

            for next_state, action, step_cost in problem.getSuccessors(current_state):
                new_g = g + step_cost
                new_f = new_g + heuristic(next_state, problem)
                new_path = path + [action]
                heapq.heappush(
                    to_visit, (new_f, new_g, new_path, next_state)
                )  # añadimos al heap el coste del next state

    return []  # no se ha encontrado solución


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
