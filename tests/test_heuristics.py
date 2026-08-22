"""启发式评估与基线棋手单元测试。"""
import unittest

from gomoku.board import Board, Move, Stone
from gomoku.heuristics import HeuristicPlayer, generate_candidates


class TestHeuristic(unittest.TestCase):
    def test_candidates_empty_is_center(self):
        b = Board(15)
        cands = generate_candidates(b)
        self.assertEqual(cands, [(7, 7)])

    def test_heuristic_takes_win(self):
        # 黑棋已有 4 连（两端均开放），任一开放端补子即成五取胜
        b = Board(15)
        for c in range(3, 7):
            b.place(Move(7, c), Stone.BLACK)
        hp = HeuristicPlayer("P")
        hp.set_stone(Stone.BLACK)
        move = hp.choose_move(b)
        # 左端 (7,2) 或右端 (7,7) 皆成五，均为正确取胜手
        self.assertIn(move, [Move(7, 2), Move(7, 7)])

    def test_heuristic_blocks_opponent_four(self):
        # 白棋有活四（两端开放），黑棋应封堵其中一端
        b = Board(15)
        for c in range(3, 7):
            b.place(Move(7, c), Stone.WHITE)
        hp = HeuristicPlayer("P")
        hp.set_stone(Stone.BLACK)
        move = hp.choose_move(b)
        self.assertIn(move, [Move(7, 2), Move(7, 7)])

    def test_heuristic_returns_legal_move(self):
        b = Board(15)
        b.place(Move(7, 7), Stone.BLACK)
        hp = HeuristicPlayer("P")
        hp.set_stone(Stone.WHITE)
        move = hp.choose_move(b)
        self.assertTrue(b.is_empty(move.row, move.col))


if __name__ == "__main__":
    unittest.main()
