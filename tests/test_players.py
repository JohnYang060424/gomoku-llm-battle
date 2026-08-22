"""棋手（含 LLM 棋手）单元测试。

重点验证 LLM 棋手的"坐标解析"与"非法重试"逻辑——即便没有真实网络，
也能确认 LLM 之间对战这条链路本身正确。
"""
import unittest
from unittest import mock

from gomoku.board import Board, Move, Stone
from gomoku import llm
from gomoku.players import ConsolePlayer, LLMPlayer, _parse_move_text


class TestParseMove(unittest.TestCase):
    def setUp(self):
        self.b = Board(15)

    def test_plain(self):
        self.assertEqual(_parse_move_text("7,7", self.b), Move(7, 7))

    def test_parenthesized(self):
        self.assertEqual(_parse_move_text("move (7,7) please", self.b), Move(7, 7))

    def test_with_spaces(self):
        self.assertEqual(_parse_move_text("7 , 7", self.b), Move(7, 7))

    def test_out_of_bounds(self):
        self.assertIsNone(_parse_move_text("20,20", self.b))

    def test_occupied(self):
        self.b.place(Move(7, 7), Stone.BLACK)
        self.assertIsNone(_parse_move_text("7,7", self.b))

    def test_garbage(self):
        self.assertIsNone(_parse_move_text("I will play somewhere", self.b))


class TestLLMPlayer(unittest.TestCase):
    def setUp(self):
        self.b = Board(15)
        self.player = LLMPlayer("LLM", "model-x", "http://x", "key")

    def test_returns_valid_move(self):
        # 用 mock 替换网络调用，验证解析与落子闭环
        with mock.patch.object(
            llm, "call_chat_completion",
            return_value=("7,7", {"content": "7,7", "reasoning": None}),
        ):
            mv = self.player.choose_move(self.b)
        self.assertEqual(mv, Move(7, 7))

    def test_retries_on_invalid_then_valid(self):
        # 第一次返回非法坐标，第二次返回合法坐标，应成功重试
        side = {"n": 0}

        def fake(*a, **k):
            side["n"] += 1
            if side["n"] == 1:
                return "99,99", {"content": "99,99", "reasoning": None}
            return "7,7", {"content": "7,7", "reasoning": None}

        with mock.patch.object(llm, "call_chat_completion", side_effect=fake):
            mv = self.player.choose_move(self.b)
        self.assertEqual(mv, Move(7, 7))
        self.assertEqual(side["n"], 2)


if __name__ == "__main__":
    unittest.main()
