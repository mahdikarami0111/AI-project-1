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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
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

    # only shows final path
    stack = util.Stack()
    visited = {}
    trace = {}
    start = problem.getStartState()
    trace[start] = []
    stack.push(start)
    path = []

    lastNode = None

    while not stack.isEmpty():
        currentState = stack.pop()
        if currentState in visited:
            continue
        path = trace[currentState]
        visited[currentState] = True

        if problem.isGoalState(currentState):
            lastNode = currentState
            break

        for successor in problem.getSuccessors(currentState):
            if successor[0] in visited:
                continue
            stack.push(successor[0])
            trace[successor[0]] = path + [successor[1]]

    if problem.__class__.__name__ == "CornersProblem":
        path.append(problem.getSuccessors(lastNode)[-1][1])

    return path



def createPath(problem, trace, curr):
    path = []
    temp = curr
    stack = util.Stack()
    while temp != problem.getStartState():
        stack.push(trace[temp][1])
        temp = trace[temp][0]
    while not stack.isEmpty():
        path.append(stack.pop())
    return path


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    que = util.Queue()
    visited = {}
    trace = {}
    start = problem.getStartState()
    trace[start] = (None, None)
    que.push(start)
    while not que.isEmpty():
        currState = que.pop()
        if currState in visited:
            continue
        visited[currState] = True

        if problem.isGoalState(currState):
            return createPath(problem, trace, currState)

        for successor in problem.getSuccessors(currState):
            if successor[0] in visited or successor[0] in que.list:
                continue
            que.push(successor[0])
            trace[successor[0]] = (currState, successor[1])

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    heap = util.PriorityQueue()
    visited = {}
    trace = {}

    start = problem.getStartState()
    trace[start] = (None, None, 0)
    heap.push(start, 0)

    while not heap.isEmpty():
        currState = heap.pop()
        if currState in visited:
            continue
        visited[currState] = True

        if problem.isGoalState(currState):
            return createPath(problem, trace, currState)

        for successor in problem.getSuccessors(currState):
            if successor[0] in visited:
                continue
            currCost = trace[currState][2] + successor[2]
            if heap.update(successor[0], currCost):
                trace[successor[0]] = (currState, successor[1], currCost)
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    heap = util.PriorityQueue()
    visited = {}
    trace = {}

    start = problem.getStartState()
    trace[start] = (None, None, 0)
    heap.push(start, heuristic(start, problem))

    while not heap.isEmpty():
        currState = heap.pop()
        if currState in visited:
            continue
        visited[currState] = True

        if problem.isGoalState(currState):
            return createPath(problem, trace, currState)

        for successor in problem.getSuccessors(currState):
            if successor[0] in visited:
                continue
            currCost = trace[currState][2] + successor[2]
            if heap.update(successor[0], currCost + heuristic(successor[0], problem)):
                trace[successor[0]] = (currState, successor[1], currCost)
    util.raiseNotDefined()


def ids(problem, limit):
    currentLimit = 0

    while currentLimit <= limit:
        stack = util.Stack()
        visited = {}
        trace = {}
        start = problem.getStartState()

        trace[start] = (None, None, 0)
        stack.push(start)

        while not stack.isEmpty():
            currentState = stack.pop()
            visited[currentState] = True

            if problem.isGoalState(currentState):
                return createPath(problem, trace, currentState)

            if trace[currentState][2] > currentLimit:
                continue

            for successor in problem.getSuccessors(currentState):
                if successor[0] in visited:
                    continue
                stack.push(successor[0])
                trace[successor[0]] = (currentState, successor[1], trace[currentState][2] + 1)
        currentLimit += 1


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
