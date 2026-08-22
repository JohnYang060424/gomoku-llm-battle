"""裁判与系列赛（三局两胜）。

裁判职责（你是裁判）：
1. 维护对局状态，按轮次向当前棋手索取落子；
2. 校验落子合法性（越界 / 重复落子由 Board 抛错，非法坐标由棋手层保证）；
3. 判定胜负（连五）或和棋；
4. 在系列赛中交替双方先后手，按"三局两胜"累计比分并提前结束；
5. 把每局结果（含落子序列、胜方、原因）记录为结构化日志，便于复盘与开源复现。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from . import config
from .board import Board, Move, Stone
from .players import Player
from .sgf import board_to_sgf


@dataclass
class GameRecord:
    """单局对局记录。"""

    game_index: int
    black_name: str
    white_name: str
    winner: Optional[str]          # "BLACK" / "WHITE" / None(和棋)
    winner_name: Optional[str]
    reason: str                    # "five-in-a-row" / "draw"
    moves: List[str] = field(default_factory=list)  # "row,col" 列表
    sgf: str = ""
    duration_ms: int = 0


@dataclass
class SeriesRecord:
    """三局两胜系列赛记录。"""

    best_of: int
    wins: dict = field(default_factory=lambda: {"BLACK": 0, "WHITE": 0})
    games: List[GameRecord] = field(default_factory=list)
    series_winner: Optional[str] = None
    series_winner_name: Optional[str] = None

    @property
    def needed(self) -> int:
        """获胜所需局数（奇数局向上取整）。"""
        return self.best_of // 2 + 1


class Referee:
    """裁判：驱动一局或一系列对局。"""

    def __init__(self, black: Player, white: Player, board_size: int = config.BOARD_SIZE) -> None:
        self.players = {Stone.BLACK: black, Stone.WHITE: white}
        black.set_stone(Stone.BLACK)
        white.set_stone(Stone.WHITE)
        self.board_size = board_size

    # ------------------------- 单局 -------------------------
    def play_game(self, game_index: int = 0) -> GameRecord:
        board = Board(self.board_size)
        start = time.time()
        moves: List[str] = []
        winner: Optional[Stone] = None
        reason = ""

        while not board.is_game_over():
            player = self.players[board.current]
            move = player.choose_move(board)
            board.place(move)  # 非法落子会抛 GomokuError
            moves.append(str(move))
            w = board.winner()
            if w is not None:
                winner = w
                reason = "five-in-a-row"
                break
            if board.is_draw():
                reason = "draw"
                break

        duration = int((time.time() - start) * 1000)
        winner_stone = winner
        winner_name = (
            self.players[winner_stone].name if winner_stone is not None else None
        )
        winner_str = winner_stone.name if winner_stone is not None else None

        rec = GameRecord(
            game_index=game_index,
            black_name=self.players[Stone.BLACK].name,
            white_name=self.players[Stone.WHITE].name,
            winner=winner_str,
            winner_name=winner_name,
            reason=reason,
            moves=moves,
            sgf=board_to_sgf(
                board,
                self.players[Stone.BLACK].name,
                self.players[Stone.WHITE].name,
                result=(winner_str or "Draw"),
            ),
            duration_ms=duration,
        )
        return rec

    # ------------------------- 系列赛（三局两胜） -------------------------
    def play_series(self, best_of: int = 3) -> SeriesRecord:
        if best_of % 2 == 0:
            raise ValueError("三局两胜等赛制局数必须为奇数")
        needed = best_of // 2 + 1
        record = SeriesRecord(best_of=best_of)
        # 捕获两名棋手的稳定引用（循环内双方会交换座位，但对象是固定的）
        p_black = self.players[Stone.BLACK]
        p_white = self.players[Stone.WHITE]
        # 按"选手"累计胜局（而非按座位色），因为双方每局交换先后手，
        # 同一座位色会被不同选手轮流占据，按座位统计会导致误判。
        wins = {p_black.name: 0, p_white.name: 0}

        for i in range(best_of):
            # 交替先后手：第 i 局让两名棋手交换黑白，保证公平
            if i % 2 == 0:
                black, white = self.players[Stone.BLACK], self.players[Stone.WHITE]
            else:
                black, white = self.players[Stone.WHITE], self.players[Stone.BLACK]
            # 重新绑定本局执子
            black.set_stone(Stone.BLACK)
            white.set_stone(Stone.WHITE)
            self.players = {Stone.BLACK: black, Stone.WHITE: white}

            game = self.play_game(game_index=i)
            record.games.append(game)

            if game.winner_name:
                wins[game.winner_name] += 1

            # 提前结束：任一方达到获胜局数
            if wins[p_black.name] >= needed:
                record.series_winner = p_black.name
                record.series_winner_name = p_black.name
                break
            if wins[p_white.name] >= needed:
                record.series_winner = p_white.name
                record.series_winner_name = p_white.name
                break

        if record.series_winner is None:
            # 打满仍未分（理论不会发生在奇数局），按胜局多者
            record.series_winner = max(wins, key=wins.get)
            record.series_winner_name = record.series_winner
        record.wins = wins
        return record

    # ------------------------- 日志 -------------------------
    def save_series_log(self, record: SeriesRecord, path: str) -> None:
        """把系列赛结构化为 JSON 写入文件（含每局 SGF，便于复盘）。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, ensure_ascii=False, indent=2)
