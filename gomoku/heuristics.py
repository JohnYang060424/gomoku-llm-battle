"""启发式评估与基线棋手。

本模块提供两件事：
1. 棋型评分工具（供基线棋手与潜在搜索引擎复用）；
2. 一个"够用且稳健"的贪心基线棋手 HeuristicPlayer，
   作为 LLM 棋手的对照基准，也用于在缺少 LLM 接口时跑通整条对战管线。

评分思路：
- 对某个候选空点，临时"放上己方棋子"，统计其四个方向形成的棋型得分（进攻分）；
- 再临时"放上对方棋子"，统计对方若在此落子的得分（防守分）；
- 先抢必胜（进攻分成五），否则优先封堵对方必胜，再按 进攻 + 防守*权重 取最优。
"""
from __future__ import annotations

import math
from typing import List, Tuple

from . import config
from .board import Board, DIRECTIONS, Move, Stone
from .players import Player  # 仅用于类型提示（players 不反向导入本模块，无循环依赖）


def _line_info(board: Board, r: int, c: int, dr: int, dc: int, stone: Stone) -> Tuple[int, int]:
    """统计以 (r,c) 为落子点的某一方向连子情况。

    前置：调用方需先临时把 board.grid[r][c] 设为 stone。
    返回 (连续同色棋子数, 两端开放数)。
    两端开放数取值 0/1/2，用于区分"活"/"眠"棋型。
    """
    count = 1  # (r,c) 自身已算 1 子
    # 正方向
    nr, nc = r + dr, c + dc
    while board.in_bounds(nr, nc) and board.grid[nr][nc] == stone:
        count += 1
        nr += dr
        nc += dc
    open_pos = board.in_bounds(nr, nc) and board.grid[nr][nc] == Stone.EMPTY
    # 反方向
    nr, nc = r - dr, c - dc
    while board.in_bounds(nr, nc) and board.grid[nr][nc] == stone:
        count += 1
        nr -= dr
        nc -= dc
    open_neg = board.in_bounds(nr, nc) and board.grid[nr][nc] == Stone.EMPTY
    return count, (1 if open_pos else 0) + (1 if open_neg else 0)


def score_pattern(count: int, open_ends: int) -> int:
    """把 (连子数, 开放端数) 映射为分值。"""
    if count >= config.WIN_COUNT:
        return config.SCORE_FIVE
    if count == 4:
        if open_ends == 2:
            return config.SCORE_OPEN_FOUR
        if open_ends == 1:
            return config.SCORE_FOUR
        return 0
    if count == 3:
        if open_ends == 2:
            return config.SCORE_OPEN_THREE
        if open_ends == 1:
            return config.SCORE_THREE
        return 0
    if count == 2:
        if open_ends == 2:
            return config.SCORE_OPEN_TWO
        if open_ends == 1:
            return config.SCORE_TWO
        return 0
    if count == 1:
        return config.SCORE_ONE if open_ends >= 1 else 0
    return 0


def evaluate_move(board: Board, r: int, c: int, stone: Stone) -> Tuple[int, int]:
    """评估在 (r,c) 落子的价值。

    返回 (进攻分, 防守分)：
    - 进攻分：己方在此落子后四个方向的棋型总分；
    - 防守分：对方若在此落子后四个方向的棋型总分（代表该点的"威胁程度"）。
    """
    # 临时落子以统计棋型，结束后还原
    board.grid[r][c] = stone
    offense = sum(
        score_pattern(*_line_info(board, r, c, dr, dc, stone))
        for dr, dc in DIRECTIONS
    )
    defense = sum(
        score_pattern(*_line_info(board, r, c, dr, dc, stone.opponent))
        for dr, dc in DIRECTIONS
    )
    board.grid[r][c] = Stone.EMPTY
    return offense, defense


def generate_candidates(board: Board) -> List[Tuple[int, int]]:
    """生成候选落点：已有棋子周围 CANDIDATE_RADIUS 范围内的空点。

    空棋盘时直接返回中心一点；否则只考虑"有邻居"的空点以大幅剪枝。
    """
    if not board.moves:
        center = board.size // 2
        return [(center, center)]
    seen = set()
    cands: List[Tuple[int, int]] = []
    rad = config.CANDIDATE_RADIUS
    for m in board.moves:
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                r, c = m.row + dr, m.col + dc
                if board.in_bounds(r, c) and board.is_empty(r, c) and (r, c) not in seen:
                    seen.add((r, c))
                    cands.append((r, c))
    return cands


def center_bias(r: int, c: int, size: int) -> float:
    """中心偏好：离中心越近得分越高，用于同分时的稳定性 tie-break。"""
    center = (size - 1) / 2
    return -(abs(r - center) + abs(c - center))


class HeuristicPlayer(Player):
    """贪心启发式基线棋手。

    策略优先级：
    1. 若某点能直接成五（进攻分成五）→ 立刻落子取胜；
    2. 否则在候选点中取 进攻 + 防守*权重 最大者；
       （对方若在某点能成五，该点防守分极高，会被自然选中封堵）
    3. 同分时取更靠近中心的点。
    """

    def __init__(self, name: str = "Heuristic", defense_weight: float = config.DEFENSE_WEIGHT) -> None:
        super().__init__(name)
        self.defense_weight = defense_weight

    def choose_move(self, board: Board) -> Move:
        if not board.moves:
            center = board.size // 2
            return Move(center, center)

        best_move: Optional[Move] = None
        best_score = -math.inf
        best_bias = -math.inf

        for (r, c) in generate_candidates(board):
            offense, defense = evaluate_move(board, r, c, self.stone)
            # 必胜优先：直接成五立即返回
            if offense >= config.SCORE_FIVE:
                return Move(r, c)
            combined = offense + defense * self.defense_weight
            bias = center_bias(r, c, board.size)
            # 先比综合分，再比中心偏好
            if combined > best_score or (combined == best_score and bias > best_bias):
                best_score, best_bias = combined, bias
                best_move = Move(r, c)

        # 理论不会走到这里（棋盘未满必有候选），兜底返回首个合法点
        if best_move is None:
            return board.legal_moves()[0]
        return best_move
