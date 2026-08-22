"""五子棋 LLM 对战框架（gomoku-llm-battle）。

核心组件：
- board       : 棋盘与规则引擎（连五判定、和棋判定）
- heuristics  : 棋型评分与启发式基线棋手
- players     : 棋手抽象 + 控制台棋手 + LLM 棋手（OpenAI 兼容）
- referee     : 裁判与三局两胜系列赛
- sgf         : SGF 存档导出
- llm         : 大模型接口封装

用法示例见 README.md；自对弈见 scripts/self_play.py。
"""
from .board import Board, GomokuError, Move, Stone
from .config import BOARD_SIZE, WIN_COUNT
from .heuristics import (
    HeuristicPlayer,
    evaluate_move,
    generate_candidates,
    score_pattern,
)
from .players import ConsolePlayer, LLMPlayer, Player
from .referee import GameRecord, Referee, SeriesRecord
from .sgf import board_to_sgf
from .search import SearchPlayer, evaluate_position, negamax, ordered_candidates

__all__ = [
    "Board", "Move", "Stone", "GomokuError",
    "BOARD_SIZE", "WIN_COUNT",
    "HeuristicPlayer", "evaluate_move", "generate_candidates", "score_pattern",
    "Player", "ConsolePlayer", "LLMPlayer",
    "Referee", "GameRecord", "SeriesRecord",
    "board_to_sgf",
    "SearchPlayer", "evaluate_position", "negamax", "ordered_candidates",
]
