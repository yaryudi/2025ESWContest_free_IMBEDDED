import sys
sys.path.append('./4PlayerChess-master/')
from actors.actor import Actor
from actors.randomStrategy import RandomStrategy
from actors.minimaxStrategy import MinimaxStrategy
from actors.evaluation import Evaluation, EvaluationV2, EvalForDepth4
from actors.moveOrdering import GlobalHistoryHeuristic, TranspositionTable

player_colors = ['r', 'b', 'y', 'g']

# Str --> Strategy:
# random: RandomStrategy
# minimax: minimaxStrategy
# none: Normal Player

# File for converting input string into Actor class objects
def generate_actors(input_strings):
  '''
  Parameters:
   - input_strings: an array of input strings
  '''
  players = []
  player_strings = input_strings[1:]
  globalHistory1 = GlobalHistoryHeuristic(12)
  globalHistory2 = GlobalHistoryHeuristic(12)
  globalTT1 = TranspositionTable()
  globalTT2 = TranspositionTable()
  for i, player in enumerate(player_strings):
    if player == 'random':
      rStrat = RandomStrategy(player_colors[i])
      actor = Actor(rStrat)
      players.append((player_colors[i], actor))
    elif player == 'minimax':
      if i % 2 == 0:
        mStrat = MinimaxStrategy(player_colors[i], 4, EvalForDepth4(), globalHistory1, globalTT1)
      else:
        mStrat = MinimaxStrategy(player_colors[i], 4, EvalForDepth4(), globalHistory2, globalTT2)
      actor = Actor(mStrat)
      players.append((player_colors[i], actor))
    elif player == 'minimax2':
      if i % 2 == 0:
        mStrat = MinimaxStrategy(player_colors[i], 1, Evaluation(), globalHistory1, None)
      else:
        mStrat = MinimaxStrategy(player_colors[i], 1, Evaluation(), globalHistory2, None)
      actor = Actor(mStrat)
      players.append((player_colors[i], actor))
  
  return players
  