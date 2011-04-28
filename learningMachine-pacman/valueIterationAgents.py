# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    self.q = util.Counter()
    states = mdp.getStates()

    for i in range (self.iterations):
      self.q = util.Counter()
      for state in states:
        actions = mdp.getPossibleActions(state)
        for action in actions:
          tAndNextState = mdp.getTransitionStatesAndProbs(state, action)
          for t in tAndNextState:
            self.q[(state, action)] = self.q[(state, action)] + (t[1] * (mdp.getReward(state, action, t[0]) + (self.discount * self.values[t[0]]))  ) 


      for state in states:
        if mdp.isTerminal(state):
          self.values[state] = 0
          continue
        actions = mdp.getPossibleActions(state)
        maxQ = -100000
        for action in actions:
          tempQ = self.q[(state, action)]
          if maxQ < tempQ:
            maxQ = tempQ
        self.values[state] = maxQ
      
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    return self.q[(state, action)]

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
      return None

    qOfTheState = util.Counter()
    acs =  self.mdp.getPossibleActions(state)
  
    for action in acs:
      tAndNextState = self.mdp.getTransitionStatesAndProbs(state, action)
      for t in tAndNextState:
        vStar = self.values[t[0]]
        qOfTheState[action] = qOfTheState[action] + (t[1] * ( self.mdp.getReward(state, action, t[0]) + self.discount * vStar) )
    return qOfTheState.argMax()
       

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
