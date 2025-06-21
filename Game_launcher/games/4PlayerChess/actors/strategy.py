import sys
import copy

sys.path.append('./4PlayerChess-master/')
from gui.board import Board
from gui.boardStruct import BoardStruct

RED, BLUE, YELLOW, GREEN, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(10)

class Strategy():
  def __init__(self, player: str):
    self.player = player
  def make_move(self, board: Board):
    # Make a move given the board state. Note, not handled in our implementation is the duplication
    # of gameboard and validation of moves from an AI actor. We just assume that these are valid and
    # pass the actual gameboard to the actor to allow them to move. 
    pass

  def promote_pawn(self, board: Board, promote_space: None or tuple):
    # given a board state, determine what to promote
    # promote space is the file, rank of the piece being promoted
    pass

  def getLegalMoves(self, board: Board, piece: int, fromFile: int, fromRank: int, player: str):
    # get legal moves for a piece given the board state. Taken from view function.
    # TODO: check that when en-passant and checks/checkmates are working that the board functions used here
    #       still work
    origin = board.square(fromFile, fromRank)
    moves = board.getSquares(board.legalMoves(piece, origin, board.colorMapping[player]) & board.emptyBB)
    captures = board.getSquares(board.legalMoves(piece, origin, board.colorMapping[player]) &
                                      board.occupiedBB)
    return moves, captures

  def getMovablePieces(self, board: Board, player: str):
    # get the pieces which can move for this player
    moveable_pieces = []
    for i, space in enumerate(board.boardData):
      if space[0] == player:
        file, rank = board.indexToFileRank(i)
        piece = board.getPiece(space)
        moves, captures = self.getLegalMoves(board, piece, file, rank, player)
        if len(moves) != 0 or len(captures) != 0:
          moveable_pieces.append((space, file, rank))
    return moveable_pieces # list of tuples. ex tuple: ('rP', 3, 2)

  def getNewBoard(self, board: Board):
    """
    Given a board, get a new board that can't be used to make moves.
    Note: this board cannot be used to make moves on the actual board, it is only meant to be used with
    search algorithms
    """
    newBoard = BoardStruct(14, 14)
    newBoard.boardData = board.boardData.copy()
    newBoard.pieceBB = copy.deepcopy(board.pieceBB)
    newBoard.occupiedBB = board.pieceBB[RED] | board.pieceBB[BLUE] | board.pieceBB[YELLOW] | board.pieceBB[GREEN]
    newBoard.emptyBB = ~newBoard.occupiedBB
    newBoard.enPassant = copy.deepcopy(board.enPassant)
    newBoard.castle = copy.deepcopy(board.castle)
    return newBoard

  def getAllLegalMoves(self, player: str, board: Board):
      movableP = self.getMovablePieces(board, player)
      allMoves = []
      allCaptures = []
      for tup in movableP:
        space, file, rank = tup
        piece = board.getPiece(space)
        moves, captures = self.getLegalMoves(board, piece, file, rank, player)
        moves = list(map(lambda x: (file, rank, *x), moves))
        captures = list(map(lambda x: (file, rank, *x), captures))
        if len(moves) != 0:
          allMoves += moves
        if len(captures) != 0:
          allCaptures += captures
      return allMoves, allCaptures

