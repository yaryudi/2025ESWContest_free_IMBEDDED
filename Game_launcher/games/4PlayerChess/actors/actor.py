
class Actor():
  def __init__(self, strategy):
    self.strategy = strategy
  def make_move(self, board):
    return self.strategy.make_move(board)
  def promote_pawn(self, board, promote_space):
    return self.strategy.promote_pawn(board, promote_space)
