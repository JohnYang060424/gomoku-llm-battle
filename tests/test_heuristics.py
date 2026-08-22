"""启发式评估与基线棋手单元测试。"""
import unittest

from gomoku.board import Board, Move, Stone
from gomoku.heuristics import HeuristicPlayer, find_threats, generate_candidates, threat_hint


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


class TestThreatDetection(unittest.TestCase):
    """威胁扫描（裁判告警）单元测试。"""

    def _place_line(self, b, stone, cells):
        for r, c in cells:
            b.place(Move(r, c), stone)

    def test_no_threat_initial(self):
        b = Board(15)
        self.assertEqual(find_threats(b, Stone.BLACK), [])
        self.assertIsNone(threat_hint(b, Stone.WHITE))

    def test_detect_open_three(self):
        # 黑棋活三 (7,5)(7,6)(7,7)，两端 (7,4)/(7,8) 空
        b = Board(15)
        self._place_line(b, Stone.BLACK, [(7, 5), (7, 6), (7, 7)])
        threats = find_threats(b, Stone.BLACK)
        self.assertEqual(len(threats), 1)
        self.assertIn("活三", threats[0])
        # 轮到白方时应收到黑方的活三告警
        hint = threat_hint(b, Stone.WHITE)
        self.assertIsNotNone(hint)
        self.assertIn("活三", hint)

    def test_detect_open_four_priority(self):
        # 白棋活四 (7,4)(7,5)(7,6)(7,7)（两端可延伸，下一手必胜）——比活三更危险
        b = Board(15)
        self._place_line(b, Stone.WHITE, [(7, 4), (7, 5), (7, 6), (7, 7)])
        # 再加一个黑活三干扰，告警应优先活四(必胜)
        self._place_line(b, Stone.BLACK, [(3, 3), (3, 4), (3, 5)])
        hint = threat_hint(b, Stone.BLACK)
        self.assertIsNotNone(hint)
        self.assertIn("必胜", hint)

    def test_duplicate_line_not_counted(self):
        # 同一活三不应被四个方向重复扫描多次
        b = Board(15)
        self._place_line(b, Stone.BLACK, [(7, 7), (8, 7), (9, 7)])  # 竖线
        threats = find_threats(b, Stone.BLACK)
        self.assertEqual(len(threats), 1)

    def test_closed_line_no_threat(self):
        # 两端被堵的三连不算威胁
        b = Board(15)
        self._place_line(b, Stone.BLACK, [(7, 5), (7, 6), (7, 7)])
        b.place(Move(7, 4), Stone.WHITE)
        b.place(Move(7, 8), Stone.WHITE)
        self.assertEqual(find_threats(b, Stone.BLACK), [])


if __name__ == "__main__":
    unittest.main()
