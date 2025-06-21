#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of the Four-Player Chess project, a four-player chess GUI.
#
# Copyright (C) 2018, GammaDeltaII
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
from PyQt5.QtCore import QObject, pyqtSignal, QSettings

# Load settings
COM = '4pc'
APP = '4PlayerChess'
SETTINGS = QSettings(COM, APP)

RED, BLUE, YELLOW, GREEN, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(10)
identifier = ['r', 'b', 'y', 'g', 'P', 'N', 'B', 'R', 'Q', 'K']

QUEENSIDE, KINGSIDE = (0, 1)

notLeftFile = 0xfffefffefffefffefffefffefffefffefffefffefffefffefffefffefffefffe
notRightFile = 0x7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff
notTopRank = 0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff

# without 3x3 corners
boardMask = 0xff00ff00ff07ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe0ff00ff00ff00000
boardEdgeMask = 0xff008100810781e400240024002400240024002781e081008100ff00000
# full 14x14 board
squareBoardMask = 0x7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe7ffe0000
squareBoardEdgeMask = 0x7ffe4002400240024002400240024002400240024002400240027ffe0000

# 256-bit De Bruijn sequence and corresponding index lookup
debruijn256 = 0x818283848586878898a8b8c8d8e8f929395969799a9b9d9e9faaeb6bedeeff
index256 = [0] * 256
for bit in range(256):
    index256[(((1 << bit) * debruijn256) >> 248) & 255] = bit


class Board(QObject):
    """The Board is the actual chess board and is the data structure shared between the View and the Algorithm."""
    boardReset = pyqtSignal()
    dataChanged = pyqtSignal(int, int)
    autoRotate = pyqtSignal(int)

    def __init__(self, files, ranks):
        super().__init__()
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

    def pieceSet(self, color, piece):
        """Gets set of pieces of one type and color."""
        return self.pieceBB[color] & self.pieceBB[piece]

    def square(self, file, rank):
        """Little-Endian Rank-File (LERF) mapping for 14x14 bitboard embedded in 16x16 bitboard (to fit 256 bits)."""
        return (rank + 1) << 4 | (file + 1)

    def square256(self, file, rank):
        """Little-Endian Rank-File (LERF) mapping for 16x16 bitboard."""
        return rank << 4 | file

    def fileRank(self, square):
        """Returns file and rank of square."""
        return (square & 15) - 1, (square >> 4) - 1

    # def bitScanForward(self, bitboard):
    #     """Finds the index of the least significant 1 bit (LS1B) using De Bruijn sequence multiplication."""
    #     assert bitboard != 0
    #     return index256[(((bitboard & -bitboard) * debruijn256) >> 248) & 255]

    def bitScanForward(self, bitboard):
        assert bitboard != 0
        return int(math.log2(bitboard & -bitboard))

    def getSquares(self, bitboard):
        """Returns list of squares (file, rank) corresponding to ones in bitboard."""
        squares = []
        while bitboard != 0:
            square = self.bitScanForward(bitboard)
            # print(square)
            # print('---')
            # print(bin(bitboard))
            # print('----')
            # print(bin(1 << square))
            squares.append(self.fileRank(square))
            bitboard ^= 1 << square
        return squares

    # def flipVertical(self, bitboard):
    #     """Flips bitboard vertically (parallel prefix approach, 4 delta swaps)."""
    #     k1 = 0x0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff
    #     k2 = 0x00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff
    #     k3 = 0x0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff
    #     bitboard = ((bitboard >> 16) & k1) | ((bitboard & k1) << 16)
    #     bitboard = ((bitboard >> 32) & k2) | ((bitboard & k2) << 32)
    #     bitboard = ((bitboard >> 64) & k3) | ((bitboard & k3) << 64)
    #     bitboard = (bitboard >> 128) | (bitboard << 128)
    #     return bitboard

    # def flipHorizontal(self, bitboard):
    #     """Flips bitboard horizontally (parallel prefix approach, 4 delta swaps)."""
    #     k1 = 0x5555555555555555555555555555555555555555555555555555555555555555
    #     k2 = 0x3333333333333333333333333333333333333333333333333333333333333333
    #     k3 = 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f
    #     k4 = 0x00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff
    #     bitboard = ((bitboard >> 1) & k1) | ((bitboard & k1) << 1)
    #     bitboard = ((bitboard >> 2) & k2) | ((bitboard & k2) << 2)
    #     bitboard = ((bitboard >> 4) & k3) | ((bitboard & k3) << 4)
    #     bitboard = ((bitboard >> 8) & k4) | ((bitboard & k4) << 8)
    #     return bitboard

    # def flipDiagonal(self, bitboard):
    #     """Flips bitboard about diagonal from lower left to upper right (parallel prefix approach, 4 delta swaps)."""
    #     k1 = 0x5555000055550000555500005555000055550000555500005555000055550000
    #     k2 = 0x3333333300000000333333330000000033333333000000003333333300000000
    #     k3 = 0x0f0f0f0f0f0f0f0f00000000000000000f0f0f0f0f0f0f0f0000000000000000
    #     k4 = 0x00ff00ff00ff00ff00ff00ff00ff00ff00000000000000000000000000000000
    #     t = k4 & (bitboard ^ (bitboard << 120))
    #     bitboard ^= t ^ (t >> 120)
    #     t = k3 & (bitboard ^ (bitboard << 60))
    #     bitboard ^= t ^ (t >> 60)
    #     t = k2 & (bitboard ^ (bitboard << 30))
    #     bitboard ^= t ^ (t >> 30)
    #     t = k1 & (bitboard ^ (bitboard << 15))
    #     bitboard ^= t ^ (t >> 15)
    #     return bitboard

    # def flipAntiDiagonal(self, bitboard):
    #     """Flips bitboard about diagonal from upper left to lower right (parallel prefix approach, 4 delta swaps)."""
    #     k1 = 0xaaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000
    #     k2 = 0xcccccccc00000000cccccccc00000000cccccccc00000000cccccccc00000000
    #     k3 = 0xf0f0f0f0f0f0f0f00000000000000000f0f0f0f0f0f0f0f00000000000000000
    #     k4 = 0xff00ff00ff00ff00ff00ff00ff00ff0000ff00ff00ff00ff00ff00ff00ff00ff
    #     t = bitboard ^ (bitboard << 136)
    #     bitboard ^= k4 & (t ^ (bitboard >> 136))
    #     t = k3 & (bitboard ^ (bitboard << 68))
    #     bitboard ^= t ^ (t >> 68)
    #     t = k2 & (bitboard ^ (bitboard << 34))
    #     bitboard ^= t ^ (t >> 34)
    #     t = k1 & (bitboard ^ (bitboard << 17))
    #     bitboard ^= t ^ (t >> 17)
    #     return bitboard

    # def rotate(self, bitboard, degrees):
    #     """Rotates bitboard +90 (clockwise), -90 (counterclockwise) or 180 degrees using two flips."""
    #     if degrees == 90:
    #         return self.flipVertical(self.flipDiagonal(bitboard))
    #     elif degrees == -90:
    #         return self.flipVertical(self.flipAntiDiagonal(bitboard))
    #     elif degrees == 180:
    #         return self.flipHorizontal(self.flipVertical(bitboard))
    #     else:
    #         pass

    def shiftN(self, bitboard, n=1):
        """Shifts bitboard north by n squares."""
        for _ in range(n):
            bitboard = (bitboard << 16) & notTopRank
        return bitboard

    def shiftNE(self, bitboard, n=1):
        """Shifts bitboard north-east by n squares."""
        for _ in range(n):
            bitboard = (bitboard << 17) & notLeftFile
        return bitboard

    def shiftE(self, bitboard, n=1):
        """Shifts bitboard east by n squares."""
        for _ in range(n):
            bitboard = (bitboard << 1) & notLeftFile
        return bitboard

    def shiftSE(self, bitboard, n=1):
        """Shifts bitboard south-east by n squares."""
        for _ in range(n):
            bitboard = (bitboard >> 15) & notRightFile
        return bitboard

    def shiftS(self, bitboard, n=1):
        """Shifts bitboard south by n squares."""
        for _ in range(n):
            bitboard >>= 16  # no wrap mask needed, as bits just fall off
        return bitboard

    def shiftSW(self, bitboard, n=1):
        """Shifts bitboard south-west by n squares."""
        for _ in range(n):
            bitboard = (bitboard >> 17) & notRightFile
        return bitboard

    def shiftW(self, bitboard, n=1):
        """Shifts bitboard west by n squares."""
        for _ in range(n):
            bitboard = (bitboard >> 1) & notRightFile
        return bitboard

    def shiftNW(self, bitboard, n=1):
        """Shifts bitboard north-west by n squares."""
        for _ in range(n):
            bitboard = (bitboard << 15) & notLeftFile
        return bitboard

    def rankMask(self, origin):
        """Returns rank passing through origin, excluding origin itself."""
        return (0xffff << (origin & 240)) ^ (1 << origin)  # excluding piece square

    def fileMask(self, origin):
        """Returns file passing through origin, excluding origin itself."""
        return (0x1000100010001000100010001000100010001000100010001000100010001 << (origin & 15)) ^ (1 << origin)

    def diagonalMask(self, origin):
        """Returns diagonal passing through origin, excluding origin itself."""
        mainDiagonal = 0x8000400020001000080004000200010000800040002000100008000400020001
        diagonal = 16 * (origin & 15) - (origin & 240)
        north = -diagonal & (diagonal >> 63)
        south = diagonal & (-diagonal >> 63)
        return ((mainDiagonal >> south) << north) ^ (1 << origin)

    def antiDiagonalMask(self, origin):
        """Returns anti-diagonal passing through origin, excluding origin itself."""
        mainDiagonal = 0x1000200040008001000200040008001000200040008001000200040008000
        diagonal = 240 - 16 * (origin & 15) - (origin & 240)
        north = -diagonal & (diagonal >> 63)
        south = diagonal & (-diagonal >> 63)
        return ((mainDiagonal >> south) << north) ^ (1 << origin)

    def rayBeyond(self, origin, square):
        """Returns part of ray from origin beyond blocker square."""
        def sign(x): return (1, -1)[x < 0]
        diff = square - origin
        s = sign(diff)
        direction = max([d if not diff % d else 1 for d in (15, 16, 17)])
        positive = -2 << square
        negative = (1 << square) - 1
        if direction == 1:
            return self.rankMask(square) & (positive if s > 0 else negative)
        elif direction == 15:
            return self.antiDiagonalMask(square) & (positive if s > 0 else negative)
        elif direction == 16:
            return self.fileMask(square) & (positive if s > 0 else negative)
        elif direction == 17:
            return self.diagonalMask(square) & (positive if s > 0 else negative)
        else:
            return 0

    def rayBetween(self, origin, square):
        """Returns part of ray from origin to square."""
        def sign(x): return (1, -1)[x < 0]
        diff = square - origin
        s = sign(diff)
        direction = max([d if not diff % d else 1 for d in (15, 16, 17)])
        posSquare = -2 << square
        negSquare = (1 << square) - 1
        posOrigin = -2 << origin
        negOrigin = (1 << origin) - 1
        if direction == 1:
            return (self.rankMask(square) & (negSquare if s > 0 else posSquare)) & \
                   (self.rankMask(origin) & (posOrigin if s > 0 else negOrigin))
        elif direction == 15:
            return (self.antiDiagonalMask(square) & (negSquare if s > 0 else posSquare)) & \
                   (self.antiDiagonalMask(origin) &
                    (posOrigin if s > 0 else negOrigin))
        elif direction == 16:
            return (self.fileMask(square) & (negSquare if s > 0 else posSquare)) & \
                   (self.fileMask(origin) & (posOrigin if s > 0 else negOrigin))
        elif direction == 17:
            return (self.diagonalMask(square) & (negSquare if s > 0 else posSquare)) & \
                   (self.diagonalMask(origin) & (posOrigin if s > 0 else negOrigin))
        else:
            return 0

    def maskBlockedSquares(self, moves, origin, occupied=None):
        """Masks blocked parts of sliding piece attack sets."""
        if not occupied:
            occupied = self.occupiedBB
        blockers = moves & occupied
        while blockers != 0:
            blockerSquare = self.bitScanForward(blockers)
            moves &= ~self.rayBeyond(origin, blockerSquare)
            blockers &= blockers - 1
        return moves

    def maskBlockedCastlingMoves(self, moves, origin, color):
        """Masks castling moves if there are pieces between the king and rook."""
        castlingMoves = moves
        while castlingMoves != 0:
            rookSquare = self.bitScanForward(castlingMoves)
            if self.rayBetween(origin, rookSquare) & self.pieceBB[color]:
                moves ^= 1 << rookSquare
            castlingMoves &= castlingMoves - 1
        return moves

    def checkMate(self, color):
        pieceTypes = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING]
        totMoves = 0
        if self.kingInCheck(color)[0]:
            for ptype in pieceTypes:
                for pieceFR in self.getSquares(self.pieceSet(color, ptype)):
                    piece = self.square(pieceFR[0], pieceFR[1])
                    totMoves = totMoves + len(self.getSquares(self.legalMoves(ptype, piece, color)))
            return totMoves == 0
        else:
            return False

    def staleMate(self, color):
        pieceTypes = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING]
        totMoves = 0
        if not self.kingInCheck(color)[0]:
            for ptype in pieceTypes:
                for pieceFR in self.getSquares(self.pieceSet(color, ptype)):
                    piece = self.square(pieceFR[0], pieceFR[1])
                    totMoves = totMoves + len(self.getSquares(self.legalMoves(ptype, piece, color)))
            return totMoves == 0
        else:
            return False




    def getProtectedSquaresAround(self, file, rank, color):
        squares = []
        pieceSet = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING]
        for piece in pieceSet:
            for square in self.getSquares(self.pieceSet(color, piece)):
                if (abs(square[0] - file) == 1 or ((square[0] - file) == 0)
                        and (abs(square[1] - rank) == 1 or ((square[1] - rank) == 0)) and ((square[0] - file) != 0 or (square[1] - rank) != 0)):
                    if not self.moreAttackersThanDefenders(square[0], square[1], color):
                        squares.append(square)
        return squares

    def getUnprotectedSquaresAround(self, file, rank, color):
        if file == 0 and rank == 0:
            squares = [(file, rank + 1), (file + 1, rank), (file + 1, rank + 1)]
        elif file == 0 and (rank != 0 or rank != 14):
            squares = [(file, rank + 1), (file + 1, rank), (file + 1, rank + 1), (file + 1, rank - 1), (file, rank - 1)]
        elif file == 0 and rank == 14:
            squares = [(file + 1, rank), (file + 1, rank - 1), (file, rank - 1)]
        elif file == 14 and rank == 0:
            squares = [(file, rank + 1), (file - 1, rank), (file - 1, rank + 1)]
        elif file == 14 and rank == 14:
            squares = [(file, rank - 1), (file - 1, rank), (file - 1, rank - 1)]
        elif file == 14 and (rank != 0 or rank != 14):
            squares = [(file, rank + 1), (file - 1, rank), (file - 1, rank + 1), (file, rank - 1), (file - 1, rank -1)]
        elif (file != 0 or file != 14) and rank == 0:
            squares = [(file - 1, rank), (file + 1, rank), (file - 1, rank + 1), (file + 1, rank + 1), (file, rank + 1)]
        elif (file != 0 or file != 14) and rank == 14:
            squares = [(file - 1, rank), (file + 1, rank), (file - 1, rank - 1), (file + 1, rank - 1), (file, rank - 1)]
        else:
            squares = [(file - 1, rank - 1), (file - 1, rank), (file - 1, rank + 1),
                       (file, rank - 1), (file, rank + 1),
                       (file + 1, rank - 1), (file + 1, rank), (file + 1, rank + 1)]
        unProtSqs = []
        for square in squares:
            if square not in self.getProtectedSquaresAround(file, rank, color):
                unProtSqs.append(square)

        return unProtSqs

    def getNumAttackedSquares(self, color):
        kingSquare = self.bitScanForward(self.pieceSet(color, KING))
        file, rank = self.fileRank(kingSquare)
        if file == 0 and rank == 0:
            squares = [(file, rank + 1), (file + 1, rank), (file + 1, rank + 1)]
        elif file == 0 and (rank != 0 or rank != 14):
            squares = [(file, rank + 1), (file + 1, rank), (file + 1, rank + 1), (file + 1, rank - 1), (file, rank - 1)]
        elif file == 0 and rank == 14:
            squares = [(file + 1, rank), (file + 1, rank - 1), (file, rank - 1)]
        elif file == 14 and rank == 0:
            squares = [(file, rank + 1), (file - 1, rank), (file - 1, rank + 1)]
        elif file == 14 and rank == 14:
            squares = [(file, rank - 1), (file - 1, rank), (file - 1, rank - 1)]
        elif file == 14 and (rank != 0 or rank != 14):
            squares = [(file, rank + 1), (file - 1, rank), (file - 1, rank + 1), (file, rank - 1), (file - 1, rank -1)]
        elif (file != 0 or file != 14) and rank == 0:
            squares = [(file - 1, rank), (file + 1, rank), (file - 1, rank + 1), (file + 1, rank + 1), (file, rank + 1)]
        elif (file != 0 or file != 14) and rank == 14:
            squares = [(file - 1, rank), (file + 1, rank), (file - 1, rank - 1), (file + 1, rank - 1), (file, rank - 1)]
        else:
            squares = [(file - 1, rank - 1), (file - 1, rank), (file - 1, rank + 1),
                       (file, rank - 1), (file, rank + 1),
                       (file + 1, rank - 1), (file + 1, rank), (file + 1, rank + 1)]
        attackers = []
        for square in squares:
            if self.moreAttackersThanDefenders(square[0], square[1], color):
                attackers.append(square)
        return len(attackers)

    def moreAttackersThanDefenders(self, file, rank, color):
        return len(self.attackersPieces(file, rank, color)) > len(self.defendersPieces(file, rank, color))

    def attackersAndDefenders(self, file, rank, color):
        return [self.attackersPieces(file, rank, color) , (self.defendersPieces(file, rank, color))]

    def attackersValue(self, file, rank, color):
        AV = 0
        attackPieces = self.attackersPieces(file, rank, color)
        for piece in attackPieces:
            if piece == PAWN:
                AV = AV + 10
            if piece == KNIGHT:
                AV = AV + 30
            if piece == BISHOP:
                AV = AV + 35
            if piece == ROOK:
                AV = AV + 50
            if piece == QUEEN:
                AV = AV + 90
        return AV

    def legalMoves(self, piece, origin, color):
        """Pseudo-legal moves for piece type."""
        if self.kingInCheck(color)[0]:
            return self.legalMovesInCheck(piece, origin, color)
        if color in (RED, YELLOW):
            friendly = self.pieceBB[RED] | self.pieceBB[YELLOW]
        else:
            friendly = self.pieceBB[BLUE] | self.pieceBB[GREEN]
        if (1 << origin) & self.absolutePins(color):
            pinMask = self.kingRay(origin, color)
        else:
            pinMask = -1
        if piece == PAWN:
            return self.pawnMoves(origin, color) & ~friendly & pinMask
        elif piece == KNIGHT:
            return self.knightMoves(origin) & ~friendly & pinMask
        elif piece == BISHOP:
            return self.maskBlockedSquares(self.bishopMoves(origin), origin) & ~friendly & pinMask
        elif piece == ROOK:
            return self.maskBlockedSquares(self.rookMoves(origin), origin) & ~friendly & pinMask
        elif piece == QUEEN:
            return self.maskBlockedSquares(self.queenMoves(origin), origin) & ~friendly & pinMask
        elif piece == KING:
            kingChecked = self.kingInCheck(color)
            if kingChecked[0]:
                castlingMoves = 0
            else:
                castlingMoves = self.castle[color][KINGSIDE] | self.castle[color][QUEENSIDE]
                moves = 0
                for move in self.getSquares((self.kingMoves(origin) & ~friendly) | self.maskBlockedCastlingMoves(castlingMoves, origin, color)):
                    if color in (RED, YELLOW):
                        # this code works because we know king is not in check
                        if not self.attacked(self.square(move[0], move[1]), BLUE) and not self.attacked(
                                self.square(move[0], move[1]), GREEN):
                            moves = moves | (1 << self.square(move[0], move[1]))
                    else:
                        if not self.attacked(self.square(move[0], move[1]), RED) and not self.attacked(
                                self.square(move[0], move[1]), YELLOW):
                            moves = moves | (1 << self.square(move[0], move[1]))
            return moves
        else:
            return -1

    def legalMovesInCheck(self, piece, origin, color):
        kingSquare = self.bitScanForward(self.pieceSet(color, KING))
        kingFile, kingRank = self.fileRank(kingSquare)
        attackersList = self.attackers(kingFile, kingRank, color)
        attackerSquares = []
        attackedSquares = []
        attackRays = []
        for att in attackersList:
            attackerSquares.append(self.square(att[0], att[1]))
        if color in (RED, YELLOW):
            friendly = self.pieceBB[RED] | self.pieceBB[YELLOW]
        else:
            friendly = self.pieceBB[BLUE] | self.pieceBB[GREEN]
        if (1 << origin) & self.absolutePins(color):
            pinMask = self.kingRay(origin, color)
        else:
            pinMask = -1

        for attack in attackerSquares:
            attackRays.append(self.attackedKingRay(attack, color))
            attackedSquares.append(self.getSquares(self.attackedKingRay(attack, color)))

        attacks = len(attackersList)

        if piece == PAWN:
            moves = 0
            for move in self.getSquares(self.pawnMoves(origin, color) & ~friendly & pinMask):
                for atPath in attackedSquares:
                    if move in atPath or move in attackersList:
                        if attacks == 1:
                            moves = moves | (1 << self.square(move[0], move[1]))
            return moves

        elif piece == KNIGHT:
            moves = 0
            for move in self.getSquares(self.knightMoves(origin) & ~friendly & pinMask):
                for atPath in attackedSquares:
                    if move in atPath or move in attackersList:
                        if attacks == 1:
                            moves = moves | (1 << self.square(move[0], move[1]))
            return moves

        elif piece == BISHOP:
            moves = 0
            for move in self.getSquares(self.maskBlockedSquares(self.bishopMoves(origin), origin) & ~friendly & pinMask):
                for atPath in attackedSquares:
                    if move in atPath or move in attackersList:
                        if attacks == 1:
                            moves = moves | (1 << self.square(move[0], move[1]))
            return moves

        elif piece == ROOK:
            moves = 0
            for move in self.getSquares(self.maskBlockedSquares(self.rookMoves(origin), origin) & ~friendly & pinMask):
                for atPath in attackedSquares:
                    if move in atPath or move in attackersList:
                        if attacks == 1:
                            moves = moves | (1 << self.square(move[0], move[1]))
            return moves

        elif piece == QUEEN:
            moves = 0
            for move in self.getSquares(self.maskBlockedSquares(self.queenMoves(origin), origin) & ~friendly & pinMask):
                for atPath in attackedSquares:
                    if move in atPath or move in attackersList:
                        if attacks == 1:
                            moves = moves | (1 << self.square(move[0], move[1]))
            return moves

        elif piece == KING:
            moves = 0
            # for king, since we always need to move this piece from its current position, we can consider it unoccupied
            # when performing a check on mask blocked squares. We flip the bit before running the calculation and then 
            # unflip to maintain code properly
            self.occupiedBB ^= 1 << origin
            for move in self.getSquares(self.maskBlockedSquares(self.kingMoves(origin), origin) & ~friendly & pinMask):
                if color in (RED, YELLOW):
                    if not self.attacked(self.square(move[0], move[1]), BLUE) and not self.attacked(self.square(move[0], move[1]), GREEN):
                        moves = moves | (1 << self.square(move[0], move[1]))
                else:
                    if not self.attacked(self.square(move[0], move[1]), RED) and not self.attacked(self.square(move[0], move[1]), YELLOW):
                        moves = moves | (1 << self.square(move[0], move[1]))
            self.occupiedBB ^= 1 << origin
            return moves

    def pawnMoves(self, origin, color, attacksOnly=False):
        """Pseudo-legal pawn moves."""
        rank4 = 0x00000000000000000000000000000000000000000000ffff0000000000000000
        rank11 = 0x0000000000000000ffff00000000000000000000000000000000000000000000
        fileD = 0x0010001000100010001000100010001000100010001000100010001000100010
        fileK = 0x0800080008000800080008000800080008000800080008000800080008000800
        origin = 1 << origin
        if color == RED:
            singlePush = self.shiftN(origin) & self.emptyBB
            doublePush = self.shiftN(singlePush) & self.emptyBB & rank4
            attacks = self.shiftNW(origin) | self.shiftNE(origin)
            captures = attacks & (
                self.pieceBB[BLUE] | self.pieceBB[GREEN] | self.enPassant[BLUE] | self.enPassant[GREEN])
        elif color == BLUE:
            singlePush = self.shiftE(origin) & self.emptyBB
            doublePush = self.shiftE(singlePush) & self.emptyBB & fileD
            attacks = self.shiftNE(origin) | self.shiftSE(origin)
            captures = attacks & (
                self.pieceBB[RED] | self.pieceBB[YELLOW] | self.enPassant[RED] | self.enPassant[YELLOW])
        elif color == YELLOW:
            singlePush = self.shiftS(origin) & self.emptyBB
            doublePush = self.shiftS(singlePush) & self.emptyBB & rank11
            attacks = self.shiftSE(origin) | self.shiftSW(origin)
            captures = attacks & (
                self.pieceBB[BLUE] | self.pieceBB[GREEN] | self.enPassant[BLUE] | self.enPassant[GREEN])
        elif color == GREEN:
            singlePush = self.shiftW(origin) & self.emptyBB
            doublePush = self.shiftW(singlePush) & self.emptyBB & fileK
            attacks = self.shiftSW(origin) | self.shiftNW(origin)
            captures = attacks & (
                self.pieceBB[RED] | self.pieceBB[YELLOW] | self.enPassant[RED] | self.enPassant[YELLOW])
        else:
            return 0
        if attacksOnly:  # only return attacked squares
            return attacks & boardMask
        else:
            return (singlePush | doublePush | captures) & boardMask

    def knightMoves(self, origin):
        """Pseudo-legal knight moves."""
        origin = 1 << origin
        NNE = self.shiftN(self.shiftNE(origin))
        NEE = self.shiftNE(self.shiftE(origin))
        SEE = self.shiftSE(self.shiftE(origin))
        SSE = self.shiftS(self.shiftSE(origin))
        SSW = self.shiftS(self.shiftSW(origin))
        SWW = self.shiftSW(self.shiftW(origin))
        NWW = self.shiftNW(self.shiftW(origin))
        NNW = self.shiftN(self.shiftNW(origin))
        return (NNE | NEE | SEE | SSE | SSW | SWW | NWW | NNW) & boardMask

    def bishopMoves(self, origin):
        """Pseudo-legal bishop moves."""
        return (self.diagonalMask(origin) | self.antiDiagonalMask(origin)) & boardMask

    def rookMoves(self, origin):
        """Pseudo-legal rook moves."""
        return (self.fileMask(origin) | self.rankMask(origin)) & boardMask

    def queenMoves(self, origin):
        """Pseudo-legal queen moves (= union of bishop and rook)."""
        return (self.bishopMoves(origin) | self.rookMoves(origin)) & boardMask

    def kingMoves(self, origin):
        """Pseudo-legal king moves."""
        kingSet = 1 << origin
        moves = self.shiftW(kingSet) | self.shiftE(kingSet)
        kingSet |= moves
        moves |= self.shiftN(kingSet) | self.shiftS(kingSet)
        return moves & boardMask

    def xrayRookAttacks(self, blockers, origin):
        """Returns X-ray rook attacks through blockers."""
        attacks = self.maskBlockedSquares(self.rookMoves(origin), origin)
        blockers &= attacks
        return attacks ^ self.maskBlockedSquares(self.rookMoves(origin), origin, self.occupiedBB ^ blockers)

    def xrayBishopAttacks(self, blockers, origin):
        """Returns X-ray rook attacks through blockers."""
        attacks = self.maskBlockedSquares(self.bishopMoves(origin), origin)
        blockers &= attacks
        return attacks ^ self.maskBlockedSquares(self.bishopMoves(origin), origin, self.occupiedBB ^ blockers)

    def absolutePins(self, color):
        """Returns absolutely (partially) pinned pieces."""
        pinned = 0
        ownPieces = self.pieceBB[color]
        kingSquare = self.bitScanForward(self.pieceSet(color, KING))
        if color in (RED, YELLOW):
            opponentRQ = self.pieceSet(BLUE, ROOK) | self.pieceSet(BLUE, QUEEN) | \
                self.pieceSet(GREEN, ROOK) | self.pieceSet(GREEN, QUEEN)
            opponentBQ = self.pieceSet(BLUE, BISHOP) | self.pieceSet(BLUE, QUEEN) | \
                self.pieceSet(GREEN, BISHOP) | self.pieceSet(GREEN, QUEEN)
        else:
            opponentRQ = self.pieceSet(RED, ROOK) | self.pieceSet(RED, QUEEN) | \
                self.pieceSet(YELLOW, ROOK) | self.pieceSet(YELLOW, QUEEN)
            opponentBQ = self.pieceSet(RED, BISHOP) | self.pieceSet(RED, QUEEN) | \
                self.pieceSet(YELLOW, BISHOP) | self.pieceSet(YELLOW, QUEEN)
        pinner = self.xrayRookAttacks(ownPieces, kingSquare) & opponentRQ
        while pinner:
            square = self.bitScanForward(pinner)
            pinned |= self.rayBetween(square, kingSquare) & ownPieces
            pinner &= pinner - 1
        pinner = self.xrayBishopAttacks(ownPieces, kingSquare) & opponentBQ
        while pinner:
            square = self.bitScanForward(pinner)
            pinned |= self.rayBetween(square, kingSquare) & ownPieces
            pinner &= pinner - 1
        return pinned

    # def aligned(self, origin, target, kingSquare):
    #     """Checks if partially pinned piece is moved along ray from or towards king."""
    #     alongRay = self.rayBetween(origin, kingSquare) & (1 << target)
    #     alongRay |= self.rayBetween(target, kingSquare) & (1 << origin)
    #     return alongRay

    def kingRay(self, square, color):
        """Returns ray from king that contains square."""
        kingSquare = self.bitScanForward(self.pieceSet(color, KING))
        return self.rayBetween(kingSquare, square) | self.rayBeyond(kingSquare, square)

    def attackedKingRay(self, square, color):
        """
        Return the ray between king and attacker
        .Added function.
        """
        kingSquare = self.bitScanForward(self.pieceSet(color, KING))
        return self.rayBetween(kingSquare, square)

    def attackers(self, file, rank, color):
        attackers = []
        kingSquareInt = self.square(file, rank)
        identifier = ['r', 'b', 'y', 'g', 'P', 'N', 'B', 'R', 'Q', 'K']
        if color in (RED, YELLOW) :
            opposite = (BLUE, GREEN)
            if self.attacked(kingSquareInt, BLUE) or self.attacked(kingSquareInt, GREEN):
              for col in opposite:
                  if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                      attackers.append(self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                  rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                  if rookMoves & self.pieceSet(col, ROOK):
                      # if rook is attacking
                      attackers.append(self.getSquares(rookMoves & self.pieceSet(col, ROOK))[0])


                  if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                      # if knight is attacking
                      attackers.append(self.getSquares(self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT))[0])

                  bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                  if bishopMoves & (self.pieceSet(col, BISHOP)):
                      # if Bishop is attacking
                      attackers.append(self.getSquares(bishopMoves & self.pieceSet(col, BISHOP))[0])

                  queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                  if queenMoves & (self.pieceSet(col, QUEEN)):
                      attackers.append(self.getSquares(queenMoves & self.pieceSet(col, QUEEN))[0])

        else:
            opposite = (RED, YELLOW)
            if self.attacked(kingSquareInt, RED) or self.attacked(kingSquareInt, YELLOW):
              for col in opposite:
                  if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                      attackers.append(
                          self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                  rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                  if rookMoves & self.pieceSet(col, ROOK):
                      # if rook is attacking
                      attackers.append(self.getSquares(rookMoves & self.pieceSet(col, ROOK))[0])

                  if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                      # if knight is attacking
                      attackers.append(
                          self.getSquares(self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT))[0])

                  bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                  if bishopMoves & (self.pieceSet(col, BISHOP)):
                      # if Bishop is attacking
                      attackers.append(self.getSquares(bishopMoves & self.pieceSet(col, BISHOP))[0])

                  queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                  if queenMoves & (self.pieceSet(col, QUEEN)):
                      attackers.append(self.getSquares(queenMoves & self.pieceSet(col, QUEEN))[0])

                  kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                  if kingMoves & self.pieceSet(col, KING):
                      attackers.append(self.getSquares(kingMoves & self.pieceSet(col, KING))[0])

        return attackers

    def attackersPieces(self, file, rank, color):
        attackers = []
        kingSquareInt = self.square(file, rank)
        if self.attackedV2(kingSquareInt, color):
           if color in (RED, YELLOW) :
                opposite = (BLUE, GREEN)
                for col in opposite:
                    if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                        attackers.append(self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                    rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                    if rookMoves & self.pieceSet(col, ROOK):
                        # if rook is attacking
                        attackers.append(ROOK)


                    if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                        # if knight is attacking
                        attackers.append(KNIGHT)

                    bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                    if bishopMoves & (self.pieceSet(col, BISHOP)):
                        # if Bishop is attacking
                        attackers.append(BISHOP)

                    queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                    if queenMoves & (self.pieceSet(col, QUEEN)):
                        attackers.append(QUEEN)

                    kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                    if kingMoves & self.pieceSet(col, KING):
                        attackers.append(KING)

           else:
               opposite = (RED, YELLOW)
               for col in opposite:
                   if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                       attackers.append(PAWN)
                   rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                   if rookMoves & self.pieceSet(col, ROOK):
                       # if rook is attacking
                       attackers.append(ROOK)

                   if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                       # if knight is attacking
                       attackers.append(KNIGHT)

                   bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                   if bishopMoves & (self.pieceSet(col, BISHOP)):
                       # if Bishop is attacking
                       attackers.append(BISHOP)

                   queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                   if queenMoves & (self.pieceSet(col, QUEEN)):
                       attackers.append(QUEEN)

                   kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                   if kingMoves & self.pieceSet(col, KING):
                       attackers.append(KING)

        return attackers

    def attackersPiecesV2(self, file, rank, color):
        attackers = []
        kingSquareInt = self.square(file, rank)
        if color in (RED, YELLOW):
            opposite = (BLUE, GREEN)
            for col in opposite:
                if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                    attackers.append(
                        self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                if rookMoves & self.pieceSet(col, ROOK):
                    # if rook is attacking
                    attackers.append(ROOK)

                if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                    # if knight is attacking
                    attackers.append(KNIGHT)

                bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                if bishopMoves & (self.pieceSet(col, BISHOP)):
                    # if Bishop is attacking
                    attackers.append(BISHOP)

                queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                if queenMoves & (self.pieceSet(col, QUEEN)):
                    attackers.append(QUEEN)

                kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                if kingMoves & self.pieceSet(col, KING):
                    attackers.append(KING)

        else:
            opposite = (RED, YELLOW)
            for col in opposite:
                if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                    attackers.append(PAWN)
                rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                if rookMoves & self.pieceSet(col, ROOK):
                    # if rook is attacking
                    attackers.append(ROOK)

                if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                    # if knight is attacking
                    attackers.append(KNIGHT)

                bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                if bishopMoves & (self.pieceSet(col, BISHOP)):
                    # if Bishop is attacking
                    attackers.append(BISHOP)

                queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                if queenMoves & (self.pieceSet(col, QUEEN)):
                    attackers.append(QUEEN)

                kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                if kingMoves & self.pieceSet(col, KING):
                    attackers.append(KING)

        return attackers

    def attacked(self, square, color):
        """Checks if a square is attacked by a player. MUST BE CALLED W/ OPPOSITE TEAM AS COLOR"""
        if color == RED:
            opposite = YELLOW
        elif color == YELLOW:
            opposite = RED
        elif color == BLUE:
            opposite = GREEN
        elif color == GREEN:
            opposite = BLUE
        else:
            return False
        if self.pawnMoves(square, opposite, True) & self.pieceSet(color, PAWN):
            return True
        if self.knightMoves(square) & self.pieceSet(color, KNIGHT):
            return True
        if self.kingMoves(square) & self.pieceSet(color, KING):
            return True
        bishopMoves = self.maskBlockedSquares(self.bishopMoves(square), square)
        if bishopMoves & (self.pieceSet(color, BISHOP) | self.pieceSet(color, QUEEN)):
            return True
        rookMoves = self.maskBlockedSquares(self.rookMoves(square), square) ^ square
        if rookMoves & (self.pieceSet(color, ROOK) | self.pieceSet(color, QUEEN)):
            return True
        return False

    def attackedV2(self, square, color):
        """Checks if a square is attacked by a player. MUST BE CALLED W/ OPPOSITE TEAM AS COLOR"""
        if color in (BLUE, GREEN):
            opposites = (RED, YELLOW)
        else:
            opposites = (BLUE, GREEN)

        for opposite in opposites:
            if self.pawnMoves(square, opposite, True) & self.pieceSet(opposite, PAWN):
                return True
            if self.knightMoves(square) & self.pieceSet(opposite, KNIGHT):
                return True
            if self.kingMoves(square) & self.pieceSet(opposite, KING):
                return True
            bishopMoves = self.maskBlockedSquares(self.bishopMoves(square), square)
            if bishopMoves & (self.pieceSet(opposite, BISHOP) | self.pieceSet(opposite, QUEEN)):
                return True
            rookMoves = self.maskBlockedSquares(self.rookMoves(square), square)
            if rookMoves & (self.pieceSet(opposite, ROOK) | self.pieceSet(opposite, QUEEN)):
                return True
        return False


    def countLegalMovesForPlayer(self, color):
        pieceTypes = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING]
        RKBQVal = []
        totMoves = 0
        for ptype in pieceTypes:
            for pieceFR in self.getSquares(self.pieceSet(color, ptype)):
                piece = self.square(pieceFR[0], pieceFR[1])
                pMoves = len(self.getSquares(self.legalMoves(ptype, piece, color)))
                RKBQVal.append(pMoves)
                totMoves = totMoves + pMoves
        return totMoves

    def countLegalMovesForPlayerV2(self, color):
        pieceTypes = [PAWN, ROOK, KNIGHT, BISHOP, QUEEN, KING]
        RBQVal = []
        totMoves = 0
        for ptype in pieceTypes:
            for pieceFR in self.getSquares(self.pieceSet(color, ptype)):
                piece = self.square(pieceFR[0], pieceFR[1])
                pMoves = len(self.getSquares(self.legalMoves(ptype, piece, color)))
                RBQVal.append(pMoves)
                totMoves = totMoves + pMoves
        return totMoves, RBQVal[1] + RBQVal[3] + RBQVal[4]


    def defenders(self, file, rank, color):
        defenders = []
        kingSquareInt = self.square(file, rank)
        if self.attackedV2(kingSquareInt, color):
           if color in (RED, YELLOW) :
                opposite = (RED, YELLOW)
                for col in opposite:
                    if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                        defenders.append(self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                    rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                    if rookMoves & self.pieceSet(col, ROOK):
                        # if rook is attacking
                        defenders.append(self.getSquares(rookMoves & self.pieceSet(col, ROOK))[0])

                    if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                        # if knight is attacking
                        defenders.append(self.getSquares(self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT))[0])

                    bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                    if bishopMoves & (self.pieceSet(col, BISHOP)):
                        # if Bishop is attacking
                        defenders.append(self.getSquares(bishopMoves & self.pieceSet(col, BISHOP))[0])

                    queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                    if queenMoves & (self.pieceSet(col, QUEEN)):
                        defenders.append(self.getSquares(queenMoves & self.pieceSet(col, QUEEN))[0])

                    kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                    if kingMoves & self.pieceSet(col, KING):
                        defenders.append(self.getSquares(kingMoves & self.pieceSet(col, KING))[0])

           else:
                opposite = (BLUE, GREEN)
                for col in opposite:

                    if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                        defenders.append(
                            self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                    rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                    if rookMoves & self.pieceSet(col, ROOK):
                        # if rook is attacking
                        defenders.append(self.getSquares(rookMoves & self.pieceSet(col, ROOK))[0])

                    if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                        # if knight is attacking
                        defenders.append(
                            self.getSquares(self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT))[0])

                    bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                    if bishopMoves & (self.pieceSet(col, BISHOP)):
                        # if Bishop is attacking
                        defenders.append(self.getSquares(bishopMoves & self.pieceSet(col, BISHOP))[0])

                    queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                    if queenMoves & (self.pieceSet(col, QUEEN)):
                        defenders.append(self.getSquares(queenMoves & self.pieceSet(col, QUEEN))[0])

                    kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                    if kingMoves & self.pieceSet(col, KING):
                        defenders.append(self.getSquares(kingMoves & self.pieceSet(col, KING))[0])

        return defenders


    def defendersPieces(self, file, rank, color):
        attackers = []
        kingSquareInt = self.square(file, rank)
        if self.attackedV2(kingSquareInt, color):
           if color in (RED, YELLOW) :
                opposite = (RED, YELLOW)
                for col in opposite:

                    if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                        attackers.append(self.getSquares(self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN))[0])
                    rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                    if rookMoves & self.pieceSet(col, ROOK):
                        # if rook is attacking
                        attackers.append(ROOK)
                    if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                        # if knight is attacking
                        attackers.append(KNIGHT)

                    bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                    if bishopMoves & (self.pieceSet(col, BISHOP)):
                        # if Bishop is attacking
                        attackers.append(BISHOP)

                    queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                    if queenMoves & (self.pieceSet(col, QUEEN)):
                        attackers.append(QUEEN)

                    kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                    if kingMoves & self.pieceSet(col, KING):
                        attackers.append(KING)

           else:
               opposite = (BLUE, GREEN)
               for col in opposite:

                   if self.pawnMoves(kingSquareInt, col, True) & self.pieceSet(col, PAWN):
                       attackers.append(PAWN)
                   rookMoves = self.maskBlockedSquares(self.rookMoves(kingSquareInt), kingSquareInt)
                   if rookMoves & self.pieceSet(col, ROOK):
                        # if rook is attacking
                        attackers.append(ROOK)

                   if self.knightMoves(kingSquareInt) & self.pieceSet(col, KNIGHT):
                        # if knight is attacking
                        attackers.append(KNIGHT)

                   bishopMoves = self.maskBlockedSquares(self.bishopMoves(kingSquareInt), kingSquareInt)
                   if bishopMoves & (self.pieceSet(col, BISHOP)):
                        # if Bishop is attacking
                        attackers.append(BISHOP)

                   queenMoves = self.maskBlockedSquares(self.queenMoves(kingSquareInt), kingSquareInt)
                   if queenMoves & (self.pieceSet(col, QUEEN)):
                       attackers.append(QUEEN)

                   kingMoves = self.maskBlockedSquares(self.kingMoves(kingSquareInt), kingSquareInt)
                   if kingMoves & self.pieceSet(col, KING):
                       attackers.append(KING)


        return attackers


    def value(self, file, rank, color):
        AV = 0
        attackPieces = self.attackersPieces(file, rank, color)
        dPieces = self.defendersPieces(file, rank, color)
        for piece in attackPieces:
            if piece == PAWN:
                AV = AV + 10
            if piece == KNIGHT:
                AV = AV + 30
            if piece == BISHOP:
                AV = AV + 35
            if piece == ROOK:
                AV = AV + 50
            if piece == QUEEN:
                AV = AV + 90
        for dPiece in dPieces:
            if dPiece == PAWN:
                AV = AV - 10
            if dPiece == KNIGHT:
                AV = AV - 30
            if dPiece == BISHOP:
                AV = AV - 35
            if dPiece == ROOK:
                AV = AV - 50
            if dPiece == QUEEN:
                AV = AV - 90
        return AV

    def expNumPieces(self, piece, color):
        squares = self.getSquares(self.pieceSet(color, piece))
        expPieces = []
        for square in squares:
            if not self.moreAttackersThanDefenders(square[0], square[1], color) and self.value(square[0], square[1], color) > 0:
                expPieces.append(square)
        return len(expPieces)





    def kingInCheck(self, color):
        """Checks if a player's king is in check."""
        kingSquare = self.bitScanForward(self.pieceSet(color, KING))
        if color in (RED, YELLOW):
            return self.attacked(kingSquare, BLUE) or self.attacked(kingSquare, GREEN), self.fileRank(kingSquare)
        else:
            return self.attacked(kingSquare, RED) or self.attacked(kingSquare, YELLOW), self.fileRank(kingSquare)

    def printBB(self, bitboard):
        """Prints 14x14 bitboard in easily readable format (for debugging)."""
        bitstring = ''
        for rank in reversed(range(14)):
            for file in range(14):
                if not ((file < 3 and rank < 3) or (file < 3 and rank > 10) or
                        (file > 10 and rank < 3) or (file > 10 and rank > 10)):
                    bitstring += '1 ' if (bitboard & (1 <<
                                          self.square(file, rank))) else '. '
                else:
                    bitstring += '  '
            bitstring += '\n'
        print(bitstring)

    def printBB256(self, bitboard):
        """Prints full 256-bit (16x16) bitboard in easily readable format (for debugging)."""
        bitstring = ''
        for rank in reversed(range(16)):
            for file in range(16):
                bitstring += '1 ' if (bitboard & (1 <<
                                      self.square256(file, rank))) else '. '
            bitstring += '\n'
        print(bitstring)

    def getPieceColor(self, char):
        """Returns piece type and color from two character identifier."""
        identifier = ['r', 'b', 'y', 'g', 'P', 'N', 'B', 'R', 'Q', 'K']
        color = identifier.index(char[0])
        piece = identifier.index(char[1])
        return piece, color

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
        self.boardReset.emit()

    def getData(self, file, rank):
        """Gets board data from square (file, rank)."""
        return self.boardData[file + rank * self.files]

    def setData(self, file, rank, data):
        """Sets board data at square (file, rank) to data."""
        index = file + rank * self.files
        if self.boardData[index] == data:
            return
        self.boardData[index] = data
        self.dataChanged.emit(file, rank)

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
                    pieceToBB = 1 << self.square(fromFile + 2, fromRank)
                    pieceToBB_ = 1 << self.square(toFile - 2, toRank)
                elif color == YELLOW and toFile < fromFile:  # kingside castle yellow
                    pieceToBB = 1 << self.square(fromFile - 2, fromRank)
                    pieceToBB_ = 1 << self.square(toFile + 2, toRank)
                elif color == BLUE and toRank > fromRank:  # kingside castle blue
                    pieceToBB = 1 << self.square(fromFile, fromRank + 2)
                    pieceToBB_ = 1 << self.square(toFile, toRank - 2)
                elif color == GREEN and toRank < fromRank:  # kingside castle green
                    pieceToBB = 1 << self.square(fromFile, fromRank - 2)
                    pieceToBB_ = 1 << self.square(toFile, toRank + 2)
                elif color == RED and toFile < fromFile:  # queenside castle red
                    pieceToBB = 1 << self.square(fromFile - 2, fromRank)
                    pieceToBB_ = 1 << self.square(toFile + 3, toRank)
                elif color == YELLOW and toFile > fromFile:  # queenside castle yellow
                    pieceToBB = 1 << self.square(fromFile + 2, fromRank)
                    pieceToBB_ = 1 << self.square(toFile - 3, toRank)
                elif color == BLUE and toRank < fromRank:  # queenside castle blue
                    pieceToBB = 1 << self.square(fromFile, fromRank - 2)
                    pieceToBB_ = 1 << self.square(toFile, toRank + 3)
                elif color == GREEN and toRank > fromRank:  # queenside castle green
                    pieceToBB = 1 << self.square(fromFile, fromRank + 2)
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

        # Emit signal for board view auto-rotation
        self.autoRotate.emit(-1)

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
        # Emit signal for board view auto-rotation
        self.autoRotate.emit(1)

    def castlingAvailability(self):
        """Returns castling availability string."""
        castling = ''
        # "K" if kingside castling available, "Q" if queenside, "-" if no player can castle
        if self.castle[RED][KINGSIDE]:
            castling += 'rK'
        if self.castle[RED][QUEENSIDE]:
            castling += 'rQ'
        if self.castle[BLUE][KINGSIDE]:
            castling += 'bK'
        if self.castle[BLUE][QUEENSIDE]:
            castling += 'bQ'
        if self.castle[YELLOW][KINGSIDE]:
            castling += 'yK'
        if self.castle[YELLOW][QUEENSIDE]:
            castling += 'yQ'
        if self.castle[GREEN][KINGSIDE]:
            castling += 'gK'
        if self.castle[GREEN][QUEENSIDE]:
            castling += 'gQ'
        if not castling:
            castling = '-'
        return castling

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
        self.boardReset.emit()

    def getFen4(self):
        """Generates FEN4 from current board state."""
        fen4 = ''
        skip = 0
        prev = ' '
        for rank in reversed(range(self.ranks)):
            for file in range(self.files):
                char = self.getData(file, rank)
                # If current square is empty, increment skip value
                if char == ' ':
                    skip += 1
                    prev = char
                else:
                    # If current square is not empty, but previous square was empty, append skip value to FEN4 string,
                    # unless the previous square was on the previous rank
                    if prev == ' ' and file != 0:
                        fen4 += str(skip)
                        skip = 0
                    # Append algebraic piece name to FEN4 string
                    fen4 += char
                    prev = char
            # If skip is non-zero at end of rank, append skip and reset to zero
            if skip > 0:
                fen4 += str(skip)
                skip = 0
            # Append slash at end of rank and append space after last rank
            if rank == 0:
                fen4 += ' '
            else:
                fen4 += '/'
        return fen4

    def getChesscomFen4(self):
        """Generates chess.com compatible FEN4."""
        fen4 = ''
        skip = 0
        prev = ' '
        for rank in reversed(range(self.ranks)):
            for file in range(self.files):
                char = self.getData(file, rank)
                # If current square is empty, increment skip value
                if char == ' ':
                    skip += 1
                    prev = char
                else:
                    # If current square is not empty, but previous square was empty, append skip value to FEN4 string,
                    # unless the previous square was on the previous rank
                    if prev == ' ' and file != 0:
                        fen4 += str(skip) + ','
                        skip = 0
                    # Append algebraic piece name to FEN4 string
                    fen4 += char + ','
                    prev = char
            # If skip is non-zero at end of rank, append skip and reset to zero
            if skip > 0:
                fen4 += str(skip) + ','
                skip = 0
            # Append slash at end of rank
            if rank != 0:
                fen4 = fen4[:-1]
                fen4 += '/'
        fen4 = fen4[:-1]
        return fen4

  # Andrew's board helper functions:
    def fileRankToIndex(self, file: int, rank: int):
        # convert file and rank into an index usable by the boardData
        return file + rank * self.files

    def indexToFileRank(self, index: int):
        # convert index (on the boardData) to file rank
        rank = index // self.files
        file = index % self.files
        return file, rank

    def getPiece(self, boardPiece: str):
        # get the relevant piece int given a str boardPiece
        return self.pieceMapping[boardPiece[1]]

    def updateEnPassant(self, piece, color, fromFile, fromRank, toFile, toRank):
        # remove any previous enPassant flags since it is a new move for this color
        self.enPassant[color] = 0
        if piece == PAWN and (abs(fromFile - toFile) == 2 or abs(fromRank - toRank) == 2):
            # add enpassant value
            offsetFile = 0
            offsetRank = 0
            if color == RED:
                offsetRank = -1
            if color == BLUE:
                offsetFile = -1
            if color == YELLOW:
                offsetRank = 1
            if color == GREEN:
                offsetFile = 1
            self.enPassant[color] = 1 << self.square(
                toFile + offsetFile, toRank + offsetRank)
