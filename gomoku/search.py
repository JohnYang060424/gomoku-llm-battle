"""搜索型棋手：负极大（negamax）+ α-β 剪枝。

定位：比纯贪心 HeuristicPlayer 更强、更接近"会下棋"的基线运动员，
能够向前看若干步、识别必胜手与双威胁。也用于演示本框架可以承载
真正有棋力的模型，而非只能跑贪心。

算法要点：
- 局面评估 evaluate_position：从我方视角近似打分（我方候选进攻分 - 对方候选进攻分）；
- 走法排序 ordered_candidates：按单点进攻+防守分降序，只保留 top-K，大幅剪枝；
- negamax 递归，叶节点返回带符号的局面分；终局（连五/和棋）直接给极端值。
"""
from __future__ import annotations

import math
from typing import List

from . import config
from .board import Board, DIRECTIONS, Move, Stone
from .heuristics import evaluate_move, generate_candidates
from .players import Player

# 终局胜负分：需远大于任何局面评分，确保搜索优先于一切局部优势
WIN_SCORE = config.SCORE_FIVE * 10


def evaluate_position(board: Board, stone: Stone) -> float:
    """从 stone 视角近似评估局面：我方所有候选进攻分 - 对方所有候选进攻分。"""
    cands = generate_candidates(board)
    my = 0
    op = 0
    for (r, c) in cands:
        off, deff = evaluate_move(board, r, c, stone)
        my += off
        op += deff
    return my - op


def ordered_candidates(board: Board, stone: Stone, top_k: int) -> List[Move]:
    """候选生成 + 排序 + 截断。

    早期棋盘候选过多时，先按中心距离裁剪到 40 个，再按 (进攻+防守) 分排序取 top-K，
    既保证相关性又控制搜索宽度。
    """
    cands = generate_candidates(board)
    if len(cands) > 40:
        center = (board.size - 1) / 2
        cands = sorted(
            cands,
            key=lambda rc: abs(rc[0] - center) + abs(rc[1] - center),
        )[:40]
    scored = []
    for (r, c) in cands:
        off, deff = evaluate_move(board, r, c, stone)
        scored.append((Move(r, c), off + deff))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [mv for mv, _ in scored[:top_k]]


def negamax(board: Board, depth: int, alpha: float, beta: float, stone: Stone) -> float:
    """负极大搜索，返回从 stone（当前轮到的一方）视角的局面价值。"""
    w = board.winner()
    if w is not None:
        return WIN_SCORE if w == stone else -WIN_SCORE
    if board.is_draw() or depth == 0:
        return evaluate_position(board, stone)

    best = -math.inf
    for mv in ordered_candidates(board, stone, config.SEARCH_TOP_K):
        board.place(mv, stone)
        val = -negamax(board, depth - 1, -beta, -alpha, stone.opponent)
        board.undo()
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:  # α-β 剪枝
            break
    return best


class SearchPlayer(Player):
    """基于 negamax + α-β 的搜索棋手。"""

    def __init__(self, name: str = "Search", depth: int = 4, top_k: int = config.SEARCH_TOP_K) -> None:
        super().__init__(name)
        self.depth = depth
        self.top_k = top_k

    def choose_move(self, board: Board) -> Move:
        if not board.moves:
            return Move(board.size // 2, board.size // 2)

        # 1) 直接取胜优先：任一候选落下即连五，立刻落子
        for mv in ordered_candidates(board, self.stone, self.top_k):
            board.place(mv, self.stone)
            win = board.winner() == self.stone
            board.undo()
            if win:
                return mv

        # 2) 否则做 negamax 搜索，选价值最高的落子
        best_move: Move = ordered_candidates(board, self.stone, self.top_k)[0]
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf
        for mv in ordered_candidates(board, self.stone, self.top_k):
            board.place(mv, self.stone)
            val = -negamax(board, self.depth - 1, -beta, -alpha, self.stone.opponent)
            board.undo()
            if val > best_val:
                best_val = val
                best_move = mv
            if val > alpha:
                alpha = val
        return best_move
