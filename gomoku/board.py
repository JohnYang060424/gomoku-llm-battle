"""棋盘与规则引擎。

职责：
1. 用二维网格表示棋盘状态；
2. 提供落子 / 悔棋 / 合法性校验；
3. 提供四方向（横、竖、两条斜线）的连五判定；
4. 提供和棋判定与棋盘渲染 / 克隆。

本文件不依赖任何第三方库，可独立测试与开源。
"""
from __future__ import annotations

import copy
import enum
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import config


class Stone(enum.IntEnum):
    """棋子颜色。使用 IntEnum 便于序列化与比较。"""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Stone":
        """返回对方棋子颜色。"""
        return Stone.WHITE if self is Stone.BLACK else Stone.BLACK

    @property
    def symbol(self) -> str:
        """用于棋盘渲染的字符。"""
        return "." if self is Stone.EMPTY else ("X" if self is Stone.BLACK else "O")


class GomokuError(Exception):
    """五子棋领域异常（越界 / 已有棋子 / 非法状态等）。"""


@dataclass(frozen=True)
class Move:
    """一个落子坐标（0 索引，row 在上、col 在右）。frozen 便于作为字典键与哈希。"""

    row: int
    col: int

    def to_index(self, size: int = config.BOARD_SIZE) -> int:
        return self.row * size + self.col

    @staticmethod
    def from_index(idx: int, size: int = config.BOARD_SIZE) -> "Move":
        return Move(idx // size, idx % size)

    def __str__(self) -> str:  # 形如 "7,7"
        return f"{self.row},{self.col}"


# 四个判定方向：水平、垂直、主对角线、副对角线
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    """五子棋棋盘。

    状态由 self.grid（size x size 的 Stone 矩阵）与落子历史 self.moves 组成。
    self.current 记录"下一步该谁落子"，由落子动作自动翻转。
    """

    def __init__(self, size: int = config.BOARD_SIZE) -> None:
        self.size = size
        self.grid: List[List[Stone]] = [
            [Stone.EMPTY] * size for _ in range(size)
        ]
        self.moves: List[Move] = []
        self.current: Stone = Stone.BLACK  # 黑棋先行（五子棋惯例）

    # ------------------------- 基础查询 -------------------------
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def is_empty(self, r: int, c: int) -> bool:
        return self.grid[r][c] == Stone.EMPTY

    def at(self, r: int, c: int) -> Stone:
        return self.grid[r][c]

    # ------------------------- 落子 / 悔棋 -------------------------
    def place(self, move: Move, stone: Optional[Stone] = None) -> None:
        """在 move 处落子。

        参数 stone 缺省时使用 self.current（即轮次自动管理）。
        非法落子（越界或已有棋子）抛出 GomokuError。
        """
        if not self.in_bounds(move.row, move.col):
            raise GomokuError(f"落子 {move} 越界（棋盘边长 {self.size}）")
        if not self.is_empty(move.row, move.col):
            raise GomokuError(f"位置 {move} 已有棋子，不能重复落子")
        stone = stone or self.current
        self.grid[move.row][move.col] = stone
        self.moves.append(move)
        self.current = stone.opponent  # 切换轮次

    def undo(self) -> None:
        """撤销上一步落子（用于搜索 / 悔棋）。"""
        if not self.moves:
            return
        last = self.moves.pop()
        self.grid[last.row][last.col] = Stone.EMPTY
        self.current = self.current.opponent

    # ------------------------- 胜负判定 -------------------------
    def check_win_at(self, move: Move, stone: Stone) -> bool:
        """判断以 move 为终点落下的 stone 是否形成连五。

        仅从 move 出发沿四个方向统计连续同色棋子数，效率为 O(1)。
        freestyle 规则下，>= WIN_COUNT 即判胜（允许"长连"获胜）。
        """
        for dr, dc in DIRECTIONS:
            count = 1
            # 正方向延伸
            r, c = move.row + dr, move.col + dc
            while self.in_bounds(r, c) and self.grid[r][c] == stone:
                count += 1
                r += dr
                c += dc
            # 反方向延伸
            r, c = move.row - dr, move.col - dc
            while self.in_bounds(r, c) and self.grid[r][c] == stone:
                count += 1
                r -= dr
                c -= dc
            if count >= config.WIN_COUNT:
                return True
        return False

    def winner(self) -> Optional[Stone]:
        """若存在胜者（最后一步形成连五）返回其颜色，否则返回 None。"""
        if not self.moves:
            return None
        last = self.moves[-1]
        stone = self.grid[last.row][last.col]
        return stone if self.check_win_at(last, stone) else None

    def is_draw(self) -> bool:
        """棋盘下满且无人连五则为和棋。"""
        return len(self.moves) == self.size * self.size and self.winner() is None

    def is_game_over(self) -> bool:
        return self.winner() is not None or self.is_draw()

    # ------------------------- 工具方法 -------------------------
    def clone(self) -> "Board":
        """深拷贝棋盘（供搜索引擎使用，不影响原状态）。"""
        new = Board(self.size)
        new.grid = copy.deepcopy(self.grid)
        new.moves = list(self.moves)
        new.current = self.current
        return new

    def legal_moves(self) -> List[Move]:
        """返回所有合法空点（用于穷举 / 校验）。"""
        return [
            Move(r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r][c] == Stone.EMPTY
        ]

    def render(self) -> str:
        """生成带坐标标尺的控制台棋盘文本。"""
        header = "   " + "".join(f"{c:2d}" for c in range(self.size))
        lines = [header]
        for r in range(self.size):
            row_str = " ".join(stone.symbol for stone in self.grid[r])
            lines.append(f"{r:2d} {row_str}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
