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
from .heuristics import threat_hint
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

    def __init__(
        self,
        black: Player,
        white: Player,
        board_size: int = config.BOARD_SIZE,
        live_log_path: Optional[str] = None,
    ) -> None:
        self.players = {Stone.BLACK: black, Stone.WHITE: white}
        black.set_stone(Stone.BLACK)
        white.set_stone(Stone.WHITE)
        self.board_size = board_size
        self.live_log_path = live_log_path  # 逐手实时落盘（JSONL），中途可监控/崩溃可挽留

    # ------------------------- 实时落盘 -------------------------
    def _live(self, rec: dict) -> None:
        """追加一条实时对局记录（JSONL）。不 buffering、直接落盘：长对局中进程崩溃
        或被人为中断，已走的每一步都留有痕迹，可监控、可复盘、可恢复。
        """
        if not self.live_log_path:
            return
        with open(self.live_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------- 单局 -------------------------
    def play_game(self, game_index: int = 0) -> GameRecord:
        board = Board(self.board_size)
        start = time.time()
        moves: List[str] = []
        winner: Optional[Stone] = None
        reason = ""
        print(f"--- 第 {game_index + 1} 局开始: 黑={self.players[Stone.BLACK].name} vs 白={self.players[Stone.WHITE].name} ---", flush=True)
        self._live({"event": "game_start", "game_index": game_index,
                    "black": self.players[Stone.BLACK].name,
                    "white": self.players[Stone.WHITE].name})

        while not board.is_game_over():
            player = self.players[board.current]
            # 机械裁判扫描对手威胁，注入提示（LLM 棋手拼进提示词，其余棋手忽略）
            hint = threat_hint(board, board.current)
            player.hint = hint
            # 威胁告警也实时落盘，复盘时能对齐"哪一步该防没防"
            if hint:
                self._live({"event": "threat", "game_index": game_index,
                            "ply": len(moves) + 1, "mover": board.current.name,
                            "hint": hint})
            move = player.choose_move(board)
            board.place(move)  # 非法落子会抛 GomokuError
            moves.append(str(move))
            # 每步进度输出（便于实时监控长对局）
            print(f"  [G{game_index + 1} M{len(moves)}] {player.name}({board.current.opponent.name}) -> {move}", flush=True)
            # 逐手实时落盘（JSONL）：进程崩溃/中断也不丢已走步骤
            self._live({"event": "move", "game_index": game_index,
                        "ply": len(moves), "stone": board.current.opponent.name,
                        "move": str(move), "player": player.name})
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
        print(f"--- 第 {game_index + 1} 局结束: 胜方={winner_name}({winner_str}), 原因={reason}, 手数={len(moves)}, 用时={duration}ms ---", flush=True)
        self._live({"event": "game_end", "game_index": game_index,
                    "winner": winner_str, "winner_name": winner_name,
                    "reason": reason, "plies": len(moves), "duration_ms": duration})

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
        # 显示名：同名（如同一模型自我对弈）时自动加（黑）/（白）后缀，
        # 保证 wins 的键唯一——否则 dict 键重合，胜局互相污染、判负提前结束。
        name_b = p_black.name
        name_w = p_white.name
        if name_b == name_w:
            name_b = f"{name_b} (黑)"
            name_w = f"{name_w} (白)"
        # 按"选手对象"累计胜局（而非按名字/座位色）：双方每局交换先后手，
        # 同一对象才是稳定的选手身份；名字可能冲突（自我对弈），id 不会。
        win_by_id = {id(p_black): 0, id(p_white): 0}
        names_by_id = {id(p_black): name_b, id(p_white): name_w}

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

            # 胜者按本局座位映射到对应选手对象（game.winner 是座位色）
            if game.winner == Stone.BLACK.name:
                win_by_id[id(black)] += 1
            elif game.winner == Stone.WHITE.name:
                win_by_id[id(white)] += 1

            # 提前结束：任一方达到获胜局数
            if win_by_id[id(p_black)] >= needed:
                record.series_winner = name_b
                record.series_winner_name = name_b
                break
            if win_by_id[id(p_white)] >= needed:
                record.series_winner = name_w
                record.series_winner_name = name_w
                break

        if record.series_winner is None:
            # 打满仍未分（理论不会发生在奇数局），按胜局多者
            winner_id = max(win_by_id, key=lambda k: win_by_id[k])
            record.series_winner = names_by_id[winner_id]
            record.series_winner_name = names_by_id[winner_id]
        record.wins = {name_b: win_by_id[id(p_black)], name_w: win_by_id[id(p_white)]}
        return record

    # ------------------------- 日志 -------------------------
    def save_series_log(self, record: SeriesRecord, path: str) -> None:
        """把系列赛结构化为 JSON 写入文件（含每局 SGF，便于复盘）。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, ensure_ascii=False, indent=2)
