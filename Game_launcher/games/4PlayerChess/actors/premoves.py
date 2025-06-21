# A file keeping a list of move opening sets for each player w/o needing to search.
import sys
import random
sys.path.append('./4PlayerChess-master/')

RED_MOVES = [
# Safe opening
[(9, 0, 8, 2), (4, 0, 5, 2), (6, 1, 6, 2), (7, 1, 7, 3), (8, 0, 7, 1), (7, 0, 10, 0)],
# Faster opening
[(9, 1, 9, 2), (8, 0, 9, 1), (9, 0, 8, 2), (7, 0, 10, 0)],
# Queenside focus opening
[(6, 1, 6, 3), (4, 0, 5, 2), (7, 1, 7, 2), (5, 0, 6, 1)],
]
BLUE_MOVES = [
# Safe opening
[(0, 9, 2, 8), (0, 4, 2, 5), (1, 6, 2, 6), (1, 7, 3, 7), (0, 8, 1, 7), (0, 7, 0, 10)],
# Faster opening
[(1, 9, 2, 9), (0, 8, 1, 9), (0, 9, 2, 8), (0, 7, 0, 10)],
# Alt opening
[(1, 6, 3, 6), (0, 4, 2, 5), (1, 7, 2, 7), (0, 5, 1, 6)],
]
YELLOW_MOVES = [
# Safe opening
[(4, 13, 5, 11), (9, 13, 8, 11), (7, 12, 7, 11), (6, 12, 6, 10), (5, 13, 6, 12), (6, 13, 3, 13)],
# Faster opening
[(4, 12, 4, 11), (5, 13, 4, 12), (4, 13, 5, 11), (6, 13, 3, 13)],
# Alt opening
[(7, 12, 7, 10), (9, 13, 8, 11), (6, 12, 6, 11), (8, 13, 7, 12)],
]
GREEN_MOVES = [
# Safe opening
[(13, 4, 11, 5), (13, 9, 11, 8), (12, 7, 11, 7), (12, 6, 10, 6), (13, 5, 12, 6), (13, 6, 13, 3)],
# Faster opening
[(12, 4, 11, 4), (13, 5, 12, 4), (13, 4, 11, 5), (13, 6, 13, 3)],
# Alt opening
[(12, 7, 10, 7), (13, 9, 11, 8), (12, 6, 11, 6), (13, 8, 12, 7)],
]


class OpeningPreMoves():
  """
  Class containing various premoved openings for 4p chess
  """
  def __init__(self) -> None:
    self.openings = {'r': RED_MOVES, 'b': BLUE_MOVES, 'y': YELLOW_MOVES, 'g': GREEN_MOVES}
  
  def getRandomOpener(self, color):
    return random.choice(self.openings[color])