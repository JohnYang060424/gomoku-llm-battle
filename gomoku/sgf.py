"""SGF（Smart Game Format）导出。

SGF 是棋类游戏通用的文本存档格式，便于在外部工具（如 Sabaki、NGS）中
复盘与共享。五子棋使用 GM[1]，坐标以字母 a..o 表示 0..14。
"""
from __future__ import annotations

from .board import Board, Move, Stone


def _sgf_coord(move: Move) -> str:
    """把 (row, col) 转为 SGF 坐标：列为首字母、行为次字母。"""
    return chr(ord("a") + move.col) + chr(ord("a") + move.row)


def board_to_sgf(
    board: Board,
    black_name: str = "Black",
    white_name: str = "White",
    result: str = "",
) -> str:
    """把整盘棋盘序列化为 SGF 字符串。"""
    head = (
        f"(;FF[4]GM[1]SZ[{board.size}]"
        f"PB[{black_name}]PW[{white_name}]"
    )
    if result:
        head += f"RE[{result}]"
    body = ""
    # 按落子顺序交替着色
    for i, mv in enumerate(board.moves):
        color = "B" if i % 2 == 0 else "W"
        body += f";{color}[{_sgf_coord(mv)}]"
    return head + body + ")"
