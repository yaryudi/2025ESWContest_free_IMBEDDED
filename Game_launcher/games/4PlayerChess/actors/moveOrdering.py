# This file is meant to be a collection of move ordering functions that can be used w/ alpha-beta pruning to 
# hopefully search the most valuable moves early (and therefore increasing # of pruned nodes)

from collections import defaultdict
import random
import sys
sys.path.append('./4PlayerChess-master/')
from gui.board import Board

# -----------------------------------------------------------------------------------------------------------
# MVV LVA Section
# -----------------------------------------------------------------------------------------------------------

# A score of capture values. indices are as follows: 
# pawn = 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5
# To use, capture_values[x][y] where x is attacker, y is victim
# for example: capture_values[4][1] = 201 which is not very good 
# (and makes sense, since queen taking rook is bad)
capture_values = [
  [105, 205, 305, 405, 505, 605],
  [104, 204, 304, 404, 504, 604],
  [103, 203, 303, 403, 503, 603],
  [102, 202, 302, 402, 502, 602],
  [101, 201, 301, 401, 501, 601],
  [100, 200, 300, 400, 500, 600]
]

pieceToCaptureValueIndexMap = {
  'P': 0,
  'N': 1,
  'B': 2,
  'R': 3,
  'Q': 4,
  'K': 5
}

def captureCoordsToValue(capture, board: Board):
  fromRank, fromFile, toRank, toFile = capture[0], capture[1], capture[2], capture[3]
  fromIndex = board.fileRankToIndex(fromRank, fromFile)
  toIndex = board.fileRankToIndex(toRank, toFile)
  attacker = board.boardData[fromIndex][1]
  victim = board.boardData[toIndex][1]
  return capture_values[pieceToCaptureValueIndexMap[attacker]][pieceToCaptureValueIndexMap[victim]]

def mvv_lva(captures, board):
  """
  Given a list of possible capture moves [tuples of the form (fromRank, fromFile, toRank, toFile)], 
  return a sorted list by most valuable victim and least valuable attacker
  """
  return sorted(captures, key=lambda x: captureCoordsToValue(x, board), reverse=True)

# -----------------------------------------------------------------------------------------------------------
# Killer Moves Heuristic Section
# -----------------------------------------------------------------------------------------------------------

class KillerMoves():
  """
  Class to hold the data structure for killer moves
  Killer moves are of the form (fromRank, fromFile, toRank, toFile)
  This class stores 2 killer moves per depth
  """
  def __init__(self, max_depth):
    self.storedMoves = []
    for _ in range(max_depth):
      self.storedMoves.append([])

  def store_move(self, move, depth):
    """
    Add a killer move (fromRank, fromFile, toRank, toFile) to the killer moves at given depth 
    """
    if len(self.storedMoves[depth]) == 2:
      self.storedMoves[depth].pop(0)
    self.storedMoves[depth].append(move)
  
  def isKillerMove(self, move, depth):
    """
    Check if a given move at a specific depth is a killer move
    """
    return move in self.storedMoves[depth]

  def sortMoves(self, moves, depth):
    sortedMoves = []
    for mv in moves:
      if self.isKillerMove(mv, depth):
        sortedMoves.insert(0, mv)
      else:
        sortedMoves.append(mv)
    return sortedMoves

# -----------------------------------------------------------------------------------------------------------
# History Heuristic Section
# -----------------------------------------------------------------------------------------------------------

class GlobalHistoryHeuristic():
  """
  Class to hold the data structure for storing good moves according to the history heuristic
  Moves are of the form (fromRank, fromFile, toRank, toFile)
  Structure: fromRank, fromFile --> toRank, toFile --> (value, global depth)
  """
  def __init__(self, refresh_age):
    self.globalDepth = 0
    self.refresh_age = refresh_age
    self.storedMoves = defaultdict(lambda: defaultdict(lambda: (0, self.globalDepth)))

  def store_move(self, move, depth):
    """
    Add a move (fromRank, fromFile, toRank, toFile) to the history 
    """
    old_score = self.storedMoves[(move[0], move[1])][(move[2], move[3])]
    if old_score[1] + self.refresh_age >= self.globalDepth:
      self.storedMoves[(move[0], move[1])][(move[2], move[3])] = (old_score[0] + 2**depth, self.globalDepth)
  
  def getHistoryHeuristic(self, move):
    """
    Check if a given move at a specific depth is a killer move
    """
    move_score = self.storedMoves[(move[0], move[1])][(move[2], move[3])]
    if move_score[1] + self.refresh_age >= self.globalDepth:
      return move_score[0]
    else:
      # move is too old since last cut, we don't trust the evaluation value anymore
      self.storedMoves[(move[0], move[1])][(move[2], move[3])] = (0, self.globalDepth)
      return 0

  def sortMoves(self, moves):
    return sorted(moves, reverse=True, key=lambda x: self.getHistoryHeuristic(x))
  
  def incrementGlobalDepth(self):
    self.globalDepth += 1

  # TODO: add function that clears out old entries
  



# -----------------------------------------------------------------------------------------------------------
# Transposition Table Section
# -----------------------------------------------------------------------------------------------------------

class TableNode():
  """
  Nodes Values for Hashes in the Transposition Table
  """
  def __init__(self, nodeType, score, numPieces, bestMove):
    """
    Create node
    Parameters:
     - nodeType: 'exact' | 'upper' | 'lower'
     - score: evaluation score
     - depth: search depth
    """
    self.nodeType = nodeType
    self.score = score
    self.numPieces = numPieces
    self.bestMove = bestMove

class TranspositionTable():
  """
  Class Holding Transposition Table for 4 Player Chess Game
  """
  def __init__(self):
    self.zTable = [[[random.randint(1, 2**64 - 1) for _ in range(24)] for _ in range(14)] for _ in range(14)]
    self.storedPositions = {}
    self.globalDepth = 0
  
  def pieceToIndex(self, piece):
    pieces = [
      'rP', 'rN', 'rB', 'rR', 'rQ', 'rK',
      'bP', 'bN', 'bB', 'bR', 'bQ', 'bK',
      'yP', 'yN', 'yB', 'yR', 'yQ', 'yK',
      'gP', 'gN', 'gB', 'gR', 'gQ', 'gK',
    ]
    return pieces.index(piece) if piece in pieces else -1
  
  def computeHash(self, board):
    """
    Given the boardData in a board, return the key hash value
    """
    h = 0
    for i in range(14):
      for j in range(14):
        index = board.fileRankToIndex(i, j)
        piece = self.pieceToIndex(board.boardData[index])
        if piece != 1:
          h ^= self.zTable[i][j][piece]
    return h
  
  # def updateHash(self, hash, boardData, fromRank, fromFile, toRank, toFile, enPassant, castling):
  #   """
  #   Given a hash representing a previous board state, return an updated hash for the new boardData
  #   Parameters:
  #    - hash: the hash representing the previous board state
  #    - boardData: the boardData list from board.py
  #    - fromRank, fromFile, toRank, toFile: the move coords
  #    - enPassant: boolean for if this move is an enpassant
  #    - castling: boolean for if this move is a castling move
  #   """
  #   movedPiece = self.pieceToIndex(boardData[fromRank][fromFile])
  #   capturedPiece = self.pieceToIndex(boardData[toRank][toFile])
  #   if castling:
  #     # remove pieces from hash
  #     hash ^= self.zTable[fromRank][fromFile][movedPiece]
  #     hash ^= self.zTable[toRank][toFile][capturedPiece]
  #     # add pieces to hash but flipped (since this is a castling move)
  #     hash ^= self.zTable[fromRank][fromFile][capturedPiece]
  #     hash ^= self.zTable[toRank][toFile][movedPiece]
  #   elif enPassant:
  #     # remove pawn from hash
  #     hash ^= self.zTable[fromRank][fromFile][movedPiece]
  #     # move pawn to new spot
  #     hash ^= self.zTable[toRank][toFile][movedPiece]
  #     # find square to remove the enpassanted pawn from
  #     offsetFile = 0
  #     offsetRank = 0
  #     pawnStr = boardData[fromRank][fromFile]
  #     if pawnStr[0] == 'r':
  #       offsetRank = 1
  #     if pawnStr[0] == 'b':
  #       offsetFile = 1
  #     if pawnStr[0] == 'y':
  #       offsetRank = -1
  #     if pawnStr[0] == 'g':
  #       offsetFile = -1
  #     # remove the enpassanted pawn
  #     removedPiece = self.pieceToIndex(boardData[fromRank + offsetRank][fromFile + offsetFile])
  #     hash ^= self.zTable[fromRank + offsetRank][fromFile + offsetFile][removedPiece]
  #   else:
  #     if capturedPiece != -1:
  #       # remove captured piece from hash
  #       hash ^= self.zTable[toRank][toFile][capturedPiece]
  #     # remove movedPiece from starting location in hash and add it to new location
  #     hash ^= self.zTable[fromRank][fromFile][movedPiece]
  #     hash ^= self.zTable[toRank][toFile][movedPiece]
  #   return hash

  def storePosition(self, hash, nodeType, score, numPieces, move):
    """
    Store the position calculations for a given board state hash
    Parameters:
     - hash: numerical hash
     - nodeType: 'PV' | 'ALL' | 'CUT'
     - score: evaluation score
     - numPieces: the number of pieces on the board
     - move: the best move
    """
    storedNode = TableNode(nodeType, score, numPieces, move)
    self.storedPositions[hash] = storedNode
  
  def getPositionCalculations(self, hash):
    """
    Return the saved node calculation for the given hashed position of the board
    """
    return self.storedPositions[hash] if hash in self.storedPositions else None
  
  def cleanTable(self, numPieces):
    """
    Remove any entries that have more pieces than the current number of pieces on the board
    """
    for k, v in list(self.storedPositions.items()):
      if v.numPieces != numPieces:
        del self.storedPositions[k]