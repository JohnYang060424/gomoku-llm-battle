"""棋盘与规则引擎单元测试。"""
import unittest

from gomoku.board import Board, GomokuError, Move, Stone


class TestWinDetection(unittest.TestCase):
    def _place_row(self, board, r, c0, count, stone):
        for i in range(count):
            board.place(Move(r, c0 + i), stone)

    def test_horizontal_five(self):
        b = Board(15)
        self._place_row(b, 7, 3, 5, Stone.BLACK)
        self.assertEqual(b.winner(), Stone.BLACK)

    def test_vertical_five(self):
        b = Board(15)
        for i in range(5):
            b.place(Move(2 + i, 10), Stone.WHITE)
        self.assertEqual(b.winner(), Stone.WHITE)

    def test_diagonal_five(self):
        b = Board(15)
        for i in range(5):
            b.place(Move(1 + i, 1 + i), Stone.BLACK)
        self.assertEqual(b.winner(), Stone.BLACK)

    def test_antidiagonal_five(self):
        b = Board(15)
        for i in range(5):
            b.place(Move(1 + i, 10 - i), Stone.BLACK)
        self.assertEqual(b.winner(), Stone.BLACK)

    def test_four_not_win(self):
        b = Board(15)
        self._place_row(b, 7, 3, 4, Stone.BLACK)
        self.assertIsNone(b.winner())

    def test_overline_wins_freestyle(self):
        # freestyle 规则下长连（>=5）也判胜
        b = Board(15)
        self._place_row(b, 7, 3, 6, Stone.BLACK)
        self.assertEqual(b.winner(), Stone.BLACK)


class TestLegality(unittest.TestCase):
    def test_occupied_raises(self):
        b = Board(15)
        b.place(Move(7, 7), Stone.BLACK)
        with self.assertRaises(GomokuError):
            b.place(Move(7, 7), Stone.WHITE)

    def test_out_of_bounds_raises(self):
        b = Board(15)
        with self.assertRaises(GomokuError):
            b.place(Move(15, 15), Stone.BLACK)

    def test_undo(self):
        b = Board(15)
        b.place(Move(7, 7), Stone.BLACK)
        self.assertEqual(b.current, Stone.WHITE)
        b.undo()
        self.assertTrue(b.is_empty(7, 7))
        self.assertEqual(b.current, Stone.BLACK)


class TestDraw(unittest.TestCase):
    def test_full_board_draw(self):
        # 3x3 棋盘无法连五，填满即和棋
        b = Board(3)
        stones = [Stone.BLACK, Stone.WHITE] * 4 + [Stone.BLACK]
        idx = 0
        for r in range(3):
            for c in range(3):
                b.place(Move(r, c), stones[idx])
                idx += 1
        self.assertTrue(b.is_draw())
        self.assertIsNone(b.winner())


if __name__ == "__main__":
    unittest.main()
