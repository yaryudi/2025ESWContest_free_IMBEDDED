import sys
sys.path.append('./4PlayerChess-master/')
from gui.board import Board
from gui.boardStruct import BoardStruct

RED, BLUE, YELLOW, GREEN, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(10)

class EvalBase():
  def evaluateBoard(self, color: int, board: Board):
    pass

class Evaluation(EvalBase):

    def evaluateBoard(self, color: int, board: Board):
        evalValue = 0
        if color in (RED, YELLOW):
            if board.countLegalMovesForPlayer(RED) == 0 or board.countLegalMovesForPlayer(YELLOW) == 0:
                return -100000
            if board.countLegalMovesForPlayer(BLUE) == 0 or board.countLegalMovesForPlayer(GREEN) == 0:
                return 100000
            evalValue = evalValue + (self.curPieceValues(RED, board) + self.curPieceValues(YELLOW, board)) - (
                    self.pieceValues(BLUE, board) + self.pieceValues(GREEN, board))
            evalValue = evalValue + (self.kingSafetyVal(BLUE, board) + self.kingSafetyVal(GREEN, board)) - (
                    self.kingSafetyVal(RED, board) + self.kingSafetyVal(YELLOW, board))
            if evalValue == 0:
                evalValue = board.countLegalMovesForPlayer(color)


        else:
            if board.countLegalMovesForPlayer(RED) == 0 or board.countLegalMovesForPlayer(YELLOW) == 0:
                return 100000
            if board.countLegalMovesForPlayer(BLUE) == 0 or board.countLegalMovesForPlayer(GREEN) == 0:
                return -100000
            evalValue = evalValue + (self.curPieceValues(BLUE, board) + self.curPieceValues(GREEN, board)) - (
                        self.pieceValues(RED, board) + self.pieceValues(YELLOW, board))
            evalValue = evalValue + (self.kingSafetyVal(RED, board) + self.kingSafetyVal(YELLOW, board)) - (
                        self.kingSafetyVal(BLUE, board) + self.kingSafetyVal(GREEN, board))
            if evalValue == 0:
                print(board.countLegalMovesForPlayer(color))
                evalValue = board.countLegalMovesForPlayer(color)

        return evalValue


    def curPieceValues(self, color: int, board: Board): # needs work
        totPVal = 0
        totPVal = totPVal + (board.expNumPieces(PAWN, color) * 10)
        totPVal = totPVal + (board.expNumPieces(KNIGHT, color) * 30)
        totPVal = totPVal + (board.expNumPieces(BISHOP, color) * 35)
        totPVal = totPVal + (board.expNumPieces(ROOK, color) * 50)
        totPVal = totPVal + (board.expNumPieces(QUEEN, color) * 90)
        return totPVal

    def pieceValues(self, color: int, board: Board):
        totPVal = 0
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, PAWN))) * 10)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, KNIGHT))) * 30)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, BISHOP))) * 35)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, ROOK))) * 50)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, QUEEN))) * 90)
        return totPVal


    # want low king saftey val, 0 = king fully protected, no attackers
    def kingSafetyVal(self, color: int, board: Board):
        #print(color)
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
        #board.printBB(board.pieceSet(color, KING))
        # print('in between')
        # print(board.bitScanForward(board.pieceSet(color, KING)))
        KSV = 0
        kingSquare = board.bitScanForward(board.pieceSet(color, KING))
        kingFile, kingRank = board.fileRank(kingSquare)

        KSV = KSV + board.attackersValue(kingFile, kingRank, color)
        # get squares around the king, if protected(occupied by friendly piece) + 0, if protected but attacked + attackers value
        # if unprotected + 10 if unprotected and attacked + 2 * attackers value
        for protSquare in board.getProtectedSquaresAround(kingFile, kingRank, color):
            KSV = KSV + board.attackersValue(protSquare[0], protSquare[1], color)

        for unProtSquare in board.getUnprotectedSquaresAround(kingFile, kingRank, color):
            KSV = KSV + 2 * board.attackersValue(unProtSquare[0], unProtSquare[1], color)
        return KSV

class EvaluationV2(EvalBase):
    def evaluateBoard(self, color: int, board: Board):
        evalValue = 0
        if color in (RED, YELLOW):
            if board.countLegalMovesForPlayer(RED) == 0 or board.countLegalMovesForPlayer(YELLOW) == 0:
                return -10000
            if board.countLegalMovesForPlayer(BLUE) == 0 or board.countLegalMovesForPlayer(GREEN) == 0:
                return 10000
            evalValue = board.countLegalMovesForPlayer(RED) + board.countLegalMovesForPlayer(YELLOW) - (
                    board.countLegalMovesForPlayer(BLUE) + board.countLegalMovesForPlayer(GREEN))
        else:
            if board.countLegalMovesForPlayer(RED) == 0 or board.countLegalMovesForPlayer(YELLOW) == 0:
                return 10000
            if board.countLegalMovesForPlayer(BLUE) == 0 or board.countLegalMovesForPlayer(GREEN) == 0:
                return -10000
            evalValue = board.countLegalMovesForPlayer(BLUE) + board.countLegalMovesForPlayer(GREEN) - (
                    board.countLegalMovesForPlayer(RED) + board.countLegalMovesForPlayer(YELLOW))
        return evalValue


class EvalForDepth4(EvalBase):


    def evaluateBoard(self, color: int, board: Board):
        evalValue = 0
        redMovesRBQ = board.countLegalMovesForPlayerV2(RED)
        yellowMovesRBQ = board.countLegalMovesForPlayerV2(YELLOW)
        blueMovesRBQ = board.countLegalMovesForPlayerV2(BLUE)
        greenMovesRBQ = board.countLegalMovesForPlayerV2(GREEN)
        redMoves = redMovesRBQ[0]
        yellowMoves = yellowMovesRBQ[0]
        blueMoves = blueMovesRBQ[0]
        greenMoves = greenMovesRBQ[0]
        moveList = [redMovesRBQ, blueMovesRBQ, yellowMovesRBQ, greenMovesRBQ]
        curMoves = moveList[color]
        if curMoves[1] == 0 and curMoves[0] > 15:
            return 0
        if color in (RED, YELLOW):
            if redMoves == 0 or yellowMoves == 0:
                return -100000
            if blueMoves == 0 or greenMoves == 0:
                return 100000

            evalValue = evalValue + (self.pieceValues(BLUE, board) + self.pieceValues(GREEN, board)) - (
                    self.pieceValues(RED, board) + self.pieceValues(YELLOW, board))
            if evalValue == 0:
                evalValue = 100 * (board.getNumAttackedSquares(BLUE) + board.getNumAttackedSquares(GREEN) - (
                        board.getNumAttackedSquares(RED) + board.getNumAttackedSquares(YELLOW)
                ))
            evalValue = evalValue + (curMoves[0] / 10)
            return evalValue + (redMoves + yellowMoves - (blueMoves + greenMoves))/2
        else:
            if redMoves == 0 or yellowMoves == 0:
                return 100000
            if blueMoves == 0 or greenMoves == 0:
                return -100000

            evalValue = evalValue + (self.pieceValues(BLUE, board) + self.pieceValues(GREEN, board)) - (
                    self.pieceValues(RED, board) + self.pieceValues(YELLOW, board))
            if evalValue == 0:
                evalValue = 100 * (board.getNumAttackedSquares(RED) + board.getNumAttackedSquares(YELLOW) - (
                        board.getNumAttackedSquares(BLUE) + board.getNumAttackedSquares(GREEN)
                ))

            evalValue = evalValue + (curMoves[0]/10)
            return evalValue + (blueMoves + greenMoves - (yellowMoves + redMoves))/2
        return evalValue


    def pieceValues(self, color: int, board: Board):
        totPVal = 0
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, PAWN))) * 10)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, KNIGHT))) * 30)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, BISHOP))) * 35)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, ROOK))) * 50)
        totPVal = totPVal + (len(board.getSquares(board.pieceSet(color, QUEEN))) * 90)
        return totPVal
