"""裁判与系列赛单元测试。"""
import unittest
import tempfile
import os

from gomoku.board import Stone
from gomoku.heuristics import HeuristicPlayer
from gomoku.referee import Referee
from gomoku.search import SearchPlayer


class TestReferee(unittest.TestCase):
    def test_single_game_ends_with_winner(self):
        # 用搜索棋手（黑）对阵贪心基线（白），应能下出决定性结果（非和棋）
        black = SearchPlayer("Black", depth=4)
        white = HeuristicPlayer("White")
        ref = Referee(black, white)
        rec = ref.play_game(0)
        self.assertIsNotNone(rec.winner)  # 搜索棋手应能强制取胜，避免和棋
        self.assertEqual(rec.reason, "five-in-a-row")
        self.assertTrue(len(rec.moves) > 0)
        self.assertTrue(rec.sgf.startswith("(;FF[4]"))

    def test_best_of_three_series(self):
        black = HeuristicPlayer("Black")
        white = HeuristicPlayer("White", defense_weight=1.2)
        ref = Referee(black, white)
        series = ref.play_series(best_of=3)
        self.assertIsNotNone(series.series_winner)
        # 系列赛应在 <=3 局内结束，且胜方达到 2 胜
        self.assertLessEqual(len(series.games), 3)
        wins = series.wins[series.series_winner]
        self.assertGreaterEqual(wins, 2)

    def test_same_name_players_series_isolation(self):
        """同名选手（如同一模型自我对弈）胜局统计不得互相污染。

        回归：v0.2.0 按名字累计 wins，同名时 dict 键重合，
        导致比分错乱与提前误判冠军；v0.2.1 改为按选手对象 id 统计，
        同名时显示名自动加 (黑)/(白) 后缀，键保持唯一。
        """
        black = HeuristicPlayer("Self")
        white = HeuristicPlayer("Self", defense_weight=1.2)  # 同名、不同对象
        ref = Referee(black, white)
        series = ref.play_series(best_of=3)
        # wins 键必须是两个（同名也拆成 (黑)/(白)）
        self.assertEqual(len(series.wins), 2)
        # 两个键都应存在且值非负
        for name, score in series.wins.items():
            self.assertGreaterEqual(score, 0)
        # 冠军名必在 wins 键中，且其胜局 >= 2（真三局两胜语义）
        self.assertIn(series.series_winner, series.wins)
        self.assertGreaterEqual(series.wins[series.series_winner], 2)
        # 若未提前结束，应恰好打满所需局数或提前一分，不会因同名误判多打/少打
        self.assertLessEqual(len(series.games), 3)

    def test_series_log_is_json_serializable(self):
        black = HeuristicPlayer("Black")
        white = HeuristicPlayer("White")
        ref = Referee(black, white)
        series = ref.play_series(best_of=3)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.json")
            ref.save_series_log(series, path)
            self.assertTrue(os.path.getsize(path) > 0)


if __name__ == "__main__":
    unittest.main()
