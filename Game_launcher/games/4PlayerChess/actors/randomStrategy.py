import sys
import random

sys.path.append('./4PlayerChess-master/')
from actors.strategy import Strategy
from gui.board import Board


class RandomStrategy(Strategy):
  def __init__(self, player):
    super().__init__(player)
  def make_move(self, board: Board):
    movableP = super().getMovablePieces(board, self.player)
    space, file, rank = random.choice(movableP)
    piece = board.getPiece(space)
    moves, captures = self.getLegalMoves(board, piece, file, rank, self.player)
    poss_moves = moves + captures
    toFile, toRank = random.choice(poss_moves)
    # print('random mover')
    # print('official move:', file, rank, toFile, toRank)
    return file, rank, toFile, toRank
  def promote_pawn(self, board: Board, promote_space: None or tuple):
    return random.choice(['N', 'B', 'Q', 'R'])



# Notes:
# Board is a bunch of strings, pieces written as 2 parts _ _, first blank is color
# [r, g, b, y] and second blank is piece [R, N, B, Q, K, P]
# board is built as expected, only note is that first few rows are red pieces, meaning
# array is down to up (if that makes any sense, see printouts below)
# file: x-coord, rank: y-coord
# print(board.boardData[:14])
# print(board.boardData[14:28])
# print(board.boardData[28:42])
# print(board.boardData[42:56])
# print(board.boardData[56:70])
# print(board.boardData[70:84])
# print(board.boardData[84:98])
# print(board.boardData[98:112])
# print(board.boardData[112:126])
# print(board.boardData[126:140])
# print(board.boardData[140:154])
# print(board.boardData[154:168])
# print(board.boardData[168:182])
# print(board.boardData[182:196])

# Other debug statements
# newBoard = super().getNewBoard(board, file, rank, toFile, toRank)
# print('--new state--')
# print(newBoard.boardData)
# print('--internals--')
# for bb in newBoard.pieceBB:
#   newBoard.printBB(bb)
# print('--a--')
# newBoard.printBB(newBoard.emptyBB)
# print('--b--')
# newBoard.printBB(newBoard.occupiedBB)
# print('--c--')
# for l in newBoard.castle:
#   for bb in l:
#     newBoard.printBB(bb)
# print('--d--')
# for bb in newBoard.enPassant:
#   newBoard.printBB(bb)

# print('--new state--')
# print(newBoard.boardData[:14])
# print(newBoard.boardData[14:28])
# print(newBoard.boardData[28:42])
# print(newBoard.boardData[42:56])
# print(newBoard.boardData[56:70])
# print(newBoard.boardData[70:84])
# print(newBoard.boardData[84:98])
# print(newBoard.boardData[98:112])
# print(newBoard.boardData[112:126])
# print(newBoard.boardData[126:140])
# print(newBoard.boardData[140:154])
# print(newBoard.boardData[154:168])
# print(newBoard.boardData[168:182])
# print(newBoard.boardData[182:196])
# print('--internals--')
# for bb in newBoard.pieceBB:
#   newBoard.printBB(bb)
# print('--a--')
# newBoard.printBB(newBoard.emptyBB)
# print('--b--')
# newBoard.printBB(newBoard.occupiedBB)
# print('--c--')
# for l in newBoard.castle:
#   for bb in l:
#     newBoard.printBB(bb)
# print('--d--')
# for bb in newBoard.enPassant:
#   newBoard.printBB(bb)