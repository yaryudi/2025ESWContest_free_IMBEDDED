import sys
sys.path.append('./4PlayerChess-master/')
from gui.board import Board
from PyQt5.QtCore import QSettings

COM = '4pc'
APP = '4PlayerChess'
SETTINGS = QSettings(COM, APP)

RED, BLUE, YELLOW, GREEN, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(10)

QUEENSIDE, KINGSIDE = (0, 1)

class BoardStruct(Board):
  """
  A board structure that only performs board calculations, but has no connection to the pyqt5 application
  """
  def __init__(self, files, ranks):
    super().__init__(files, ranks)
    self.files = files
    self.ranks = ranks
    self.boardData = []
    self.pieceBB = []
    self.emptyBB = 0
    self.occupiedBB = 0
    self.castle = []
    self.enPassant = []
    self.initBoard()
    self.colorMapping = {'r': 0, 'b': 1, 'y': 2, 'g': 3}
    self.pieceMapping = {'P': 4, 'N': 5, 'B': 6, 'R': 7, 'Q': 8, 'K': 9}
  
  def initBoard(self):
    """Initializes board with empty squares."""
    self.boardData = [' '] * self.files * self.ranks
    self.pieceBB = [0] * 10
    self.emptyBB = 0
    self.occupiedBB = 0
    self.castle = [[1 << self.square(3, 0), 1 << self.square(10, 0)],
                    [1 << self.square(0, 3), 1 << self.square(0, 10)],
                    [1 << self.square(10, 13), 1 << self.square(3, 13)],
                    [1 << self.square(13, 10), 1 << self.square(13, 3)]]
    # bitmaps keeping track of when players move their pawn 2 steps (for enpassant)
    self.enPassant = [0, 0, 0, 0]
    self.castlingAvailability()
  
  def setData(self, file, rank, data):
    """Sets board data at square (file, rank) to data."""
    index = file + rank * self.files
    if self.boardData[index] == data:
        return
    self.boardData[index] = data

  def makeMove(self, fromFile, fromRank, toFile, toRank):
    """Moves piece from square (fromFile, fromRank) to square (toFile, toRank)."""
    char = self.getData(fromFile, fromRank)
    captured = self.getData(toFile, toRank)
    # If castling move, move king and rook to castling squares instead of ordinary move
    move = char + ' ' + chr(fromFile + 97) + str(fromRank + 1) + ' ' + \
        captured + ' ' + chr(toFile + 97) + str(toRank + 1)

    # check for enpassant
    if char[1] == "P" and captured == " " and abs(fromFile - toFile) == 1 and abs(fromRank - toRank) == 1:
      color = char[0]
      offsetFile = 0
      offsetRank = 0
      if color == 'r':
          offsetRank = 1
      if color == 'b':
          offsetFile = 1
      if color == 'y':
          offsetRank = -1
      if color == 'g':
          offsetFile = -1
      self.setData(toFile, toRank, char)
      self.setData(fromFile, fromRank, ' ')
      self.setData(fromFile + offsetFile, fromRank + offsetRank, ' ')
    elif move == 'rK h1 rR k1':  # kingside castle red
      self.setData(fromFile + 2, fromRank, char)
      self.setData(toFile - 2, toRank, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[RED][KINGSIDE] = 0
    elif move == 'yK g14 yR d14':  # kingside castle yellow
      self.setData(fromFile - 2, fromRank, char)
      self.setData(toFile + 2, toRank, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[YELLOW][KINGSIDE] = 0
    elif move == 'bK a8 bR a11':  # kingside castle blue
      self.setData(fromFile, fromRank + 2, char)
      self.setData(toFile, toRank - 2, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[BLUE][KINGSIDE] = 0
    elif move == 'gK n7 gR n4':  # kingside castle green
      self.setData(fromFile, fromRank - 2, char)
      self.setData(toFile, toRank + 2, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[GREEN][KINGSIDE] = 0
    elif move == 'rK h1 rR d1':  # queenside castle red
      self.setData(fromFile - 2, fromRank, char)
      self.setData(toFile + 3, toRank, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[RED][QUEENSIDE] = 0
    elif move == 'yK g14 yR k14':  # queenside castle yellow
      self.setData(fromFile + 2, fromRank, char)
      self.setData(toFile - 3, toRank, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[YELLOW][QUEENSIDE] = 0
    elif move == 'bK a8 bR a4':  # queenside castle blue
      self.setData(fromFile, fromRank - 2, char)
      self.setData(toFile, toRank + 3, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[BLUE][QUEENSIDE] = 0
    elif move == 'gK n7 gR n11':  # queenside castle green
      self.setData(fromFile, fromRank + 2, char)
      self.setData(toFile, toRank - 3, captured)
      self.setData(fromFile, fromRank, ' ')
      self.setData(toFile, toRank, ' ')
      self.castle[GREEN][QUEENSIDE] = 0
    else:  # regular move
      self.setData(toFile, toRank, char)
      self.setData(fromFile, fromRank, ' ')
      # If king move or rook move from original square, remove castling availability
      if char == 'rK' and (fromFile, fromRank) == (7, 0):
        self.castle[RED][KINGSIDE] = 0
        self.castle[RED][QUEENSIDE] = 0
      if char == 'rR' and (fromFile, fromRank) == (10, 0):
        self.castle[RED][KINGSIDE] = 0
      if char == 'rR' and (fromFile, fromRank) == (3, 0):
        self.castle[RED][QUEENSIDE] = 0
      if char == 'bK' and (fromFile, fromRank) == (0, 7):
        self.castle[BLUE][KINGSIDE] = 0
        self.castle[BLUE][QUEENSIDE] = 0
      if char == 'bR' and (fromFile, fromRank) == (0, 10):
        self.castle[BLUE][KINGSIDE] = 0
      if char == 'bR' and (fromFile, fromRank) == (0, 3):
        self.castle[BLUE][QUEENSIDE] = 0
      if char == 'yK' and (fromFile, fromRank) == (6, 13):
        self.castle[YELLOW][KINGSIDE] = 0
        self.castle[YELLOW][QUEENSIDE] = 0
      if char == 'yR' and (fromFile, fromRank) == (3, 13):
        self.castle[YELLOW][KINGSIDE] = 0
      if char == 'yR' and (fromFile, fromRank) == (10, 13):
        self.castle[YELLOW][QUEENSIDE] = 0
      if char == 'gK' and (fromFile, fromRank) == (13, 6):
        self.castle[GREEN][KINGSIDE] = 0
        self.castle[GREEN][QUEENSIDE] = 0
      if char == 'gR' and (fromFile, fromRank) == (13, 3):
        self.castle[GREEN][KINGSIDE] = 0
      if char == 'gR' and (fromFile, fromRank) == (13, 10):
        self.castle[GREEN][QUEENSIDE] = 0
    # Update bitboards
    piece, color = self.getPieceColor(char)
    self.updateEnPassant(piece, color, fromFile, fromRank, toFile, toRank)
    fromBB = 1 << self.square(fromFile, fromRank)
    toBB = 1 << self.square(toFile, toRank)
    fromToBB = fromBB ^ toBB

    # Move piece
    self.pieceBB[color] ^= fromToBB
    self.pieceBB[piece] ^= fromToBB
    self.occupiedBB ^= fromToBB
    self.emptyBB ^= fromToBB
    if captured != ' ':
        piece_, color_ = self.getPieceColor(captured)
        if piece == KING and piece_ == ROOK and color == color_:
            # Undo king move
            self.pieceBB[color] ^= fromToBB
            self.pieceBB[piece] ^= fromToBB
            self.occupiedBB ^= fromToBB
            self.emptyBB ^= fromToBB
            # Move king and rook to proper castling squares
            pieceFromBB = 1 << self.square(fromFile, fromRank)
            pieceFromBB_ = 1 << self.square(toFile, toRank)
            if color == RED and toFile > fromFile:  # kingside castle red
                pieceToBB = 1 << self.square(toFile + 2, toRank)
                pieceToBB_ = 1 << self.square(toFile - 2, toRank)
            elif color == YELLOW and toFile < fromFile:  # kingside castle yellow
                pieceToBB = 1 << self.square(toFile - 2, toRank)
                pieceToBB_ = 1 << self.square(toFile + 2, toRank)
            elif color == BLUE and toRank > fromRank:  # kingside castle blue
                pieceToBB = 1 << self.square(toFile, toRank + 2)
                pieceToBB_ = 1 << self.square(toFile, toRank - 2)
            elif color == GREEN and toRank < fromRank:  # kingside castle green
                pieceToBB = 1 << self.square(toFile, toRank - 2)
                pieceToBB_ = 1 << self.square(toFile, toRank + 2)
            elif color == RED and toFile < fromFile:  # queenside castle red
                pieceToBB = 1 << self.square(toFile - 2, toRank)
                pieceToBB_ = 1 << self.square(toFile + 3, toRank)
            elif color == YELLOW and toFile > fromFile:  # queenside castle yellow
                pieceToBB = 1 << self.square(toFile + 2, toRank)
                pieceToBB_ = 1 << self.square(toFile - 3, toRank)
            elif color == BLUE and toRank < fromRank:  # queenside castle blue
                pieceToBB = 1 << self.square(toFile, toRank - 2)
                pieceToBB_ = 1 << self.square(toFile, toRank + 3)
            elif color == GREEN and toRank > fromRank:  # queenside castle green
                pieceToBB = 1 << self.square(toFile, toRank + 2)
                pieceToBB_ = 1 << self.square(toFile, toRank - 3)
            else:  # invalid move
                pieceToBB = 0
                pieceToBB_ = 0
            pieceFromToBB = pieceFromBB ^ pieceToBB
            pieceFromToBB_ = pieceFromBB_ ^ pieceToBB_
            # Move king
            self.pieceBB[color] ^= pieceFromToBB
            self.pieceBB[piece] ^= pieceFromToBB
            self.occupiedBB ^= pieceFromToBB
            self.emptyBB ^= pieceFromToBB
            # Move rook
            self.pieceBB[color_] ^= pieceFromToBB_
            self.pieceBB[piece_] ^= pieceFromToBB_
            self.occupiedBB ^= pieceFromToBB_
            self.emptyBB ^= pieceFromToBB_
            # Undo remove captured piece (in advance)
            self.pieceBB[color_] ^= toBB
            self.pieceBB[piece_] ^= toBB
            self.occupiedBB ^= toBB
            self.emptyBB ^= toBB
        # Remove captured piece
        self.pieceBB[color_] ^= toBB
        self.pieceBB[piece_] ^= toBB
        self.occupiedBB ^= toBB
        self.emptyBB ^= toBB

  def undoMove(self, fromFile, fromRank, toFile, toRank, char, captured):
      """Takes back move and restores captured piece."""
      # Remove king and rook from castling squares
      move = char + ' ' + chr(fromFile + 97) + str(fromRank + 1) + ' ' + \
          captured + ' ' + chr(toFile + 97) + str(toRank + 1)
      if move == 'rK h1 rR k1':  # kingside castle red
          self.setData(fromFile + 2, fromRank, ' ')
          self.setData(toFile - 2, toRank, ' ')
          self.castle[RED][KINGSIDE] = 1 << self.square(10, 0)
      elif move == 'yK g14 yR d14':  # kingside castle yellow
          self.setData(fromFile - 2, fromRank, ' ')
          self.setData(toFile + 2, toRank, ' ')
          self.castle[YELLOW][KINGSIDE] = 1 << self.square(3, 13)
      elif move == 'bK a8 bR a11':  # kingside castle blue
          self.setData(fromFile, fromRank + 2, ' ')
          self.setData(toFile, toRank - 2, ' ')
          self.castle[BLUE][KINGSIDE] = 1 << self.square(0, 10)
      elif move == 'gK n7 gR n4':  # kingside castle green
          self.setData(fromFile, fromRank - 2, ' ')
          self.setData(toFile, toRank + 2, ' ')
          self.castle[GREEN][KINGSIDE] = 1 << self.square(13, 3)
      elif move == 'rK h1 rR d1':  # queenside castle red
          self.setData(fromFile - 2, fromRank, ' ')
          self.setData(toFile + 3, toRank, ' ')
          self.castle[RED][QUEENSIDE] = 1 << self.square(3, 0)
      elif move == 'yK g14 yR k14':  # queenside castle yellow
          self.setData(fromFile + 2, fromRank, ' ')
          self.setData(toFile - 3, toRank, ' ')
          self.castle[YELLOW][QUEENSIDE] = 1 << self.square(10, 13)
      elif move == 'bK a8 bR a4':  # queenside castle blue
          self.setData(fromFile, fromRank - 2, ' ')
          self.setData(toFile, toRank + 3, ' ')
          self.castle[BLUE][QUEENSIDE] = 1 << self.square(0, 3)
      elif move == 'gK n7 gR n11':  # queenside castle green
          self.setData(fromFile, fromRank + 2, ' ')
          self.setData(toFile, toRank - 3, ' ')
          self.castle[GREEN][QUEENSIDE] = 1 << self.square(13, 10)
      # Move piece back and restore captured piece
      self.setData(fromFile, fromRank, char)
      self.setData(toFile, toRank, captured)
      # Update bitboards
      piece, color = self.getPieceColor(char)
      fromBB = 1 << self.square(toFile, toRank)
      toBB = 1 << self.square(fromFile, fromRank)
      fromToBB = fromBB ^ toBB
      # Move piece back
      self.pieceBB[color] ^= fromToBB
      self.pieceBB[piece] ^= fromToBB
      self.occupiedBB ^= fromToBB
      self.emptyBB ^= fromToBB
      if captured != ' ':
          piece_, color_ = self.getPieceColor(captured)
          if piece == KING and piece_ == ROOK and color == color_:
              # Remove king and rook from castling squares
              castlingSquares = self.rayBetween(self.square(
                  fromFile, fromRank), self.square(toFile, toRank))
              self.pieceBB[color] &= ~castlingSquares
              self.pieceBB[piece_] &= ~castlingSquares
              self.pieceBB[piece] &= ~castlingSquares
              # Undo restore captured piece (in advance)
              self.pieceBB[color_] ^= fromBB
              self.pieceBB[piece_] ^= fromBB
              self.occupiedBB ^= fromBB
              self.emptyBB ^= fromBB
          # Restore captured piece
          self.pieceBB[color_] ^= fromBB
          self.pieceBB[piece_] ^= fromBB
          self.occupiedBB ^= fromBB
          self.emptyBB ^= fromBB

  def parseFen4(self, fen4):
      """Sets board position according to the FEN4 string fen4."""
      if SETTINGS.value('chesscom'):
          # Remove chess.com prefix and commas
          i = fen4.rfind('-')
          fen4 = fen4[i+1:]
          fen4 = fen4.replace(',', '')
          fen4 += ' '
      index = 0
      skip = 0
      for rank in reversed(range(self.ranks)):
          for file in range(self.files):
              if skip > 0:
                  char = ' '
                  skip -= 1
              else:
                  # Pieces are always two characters, skip value can be single or double digit
                  char = fen4[index]
                  index += 1
                  if char.isdigit():
                      # Check if next is also digit. If yes, treat as single number
                      next_ = fen4[index]
                      if next_.isdigit():
                          char += next_
                          index += 1
                      skip = int(char)
                      char = ' '
                      skip -= 1
                  # If not digit, then it is a two-character piece. Add next character
                  else:
                      char += fen4[index]
                      index += 1
              self.setData(file, rank, char)
              # Set bitboards
              if char != ' ':
                  piece, color = self.getPieceColor(char)
                  self.pieceBB[color] |= 1 << self.square(file, rank)
                  self.pieceBB[piece] |= 1 << self.square(file, rank)
          next_ = fen4[index]
          if next_ != '/' and next_ != ' ':
              # If no slash or space after rank, the FEN4 is invalid, so reset board
              self.initBoard()
              return
          else:  # Skip the slash
              index += 1
      self.occupiedBB = self.pieceBB[RED] | self.pieceBB[BLUE] | self.pieceBB[YELLOW] | self.pieceBB[GREEN]
      self.emptyBB = ~self.occupiedBB

  # def checkMate(self, color):
  #   rv = super().checkMate(color)
  #   return rv