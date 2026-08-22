#!/usr/bin/env python3
"""自对弈 / 对战运行器（三局两胜）。

用法示例：
  # 两个启发式棋手对战（用于验证管线与基准强度）
  python scripts/self_play.py --black heuristic:1.0 --white heuristic:1.2 --best-of 3

  # 搜索型棋手（更强基线）对阵贪心基线
  python scripts/self_play.py --black search:4 --white heuristic:1.0 --best-of 3

  # 人工亲自上场（控制台输入），对手为启发式
  python scripts/self_play.py --black console --white heuristic:1.0

  # 单个 LLM 自我对弈（同一模型，一黑一白）
  python scripts/self_play.py --black llm:gpt-4o --white llm:gpt-4o

  # 任意两个 LLM 互搏（含跨服务商）：内联各自端点
  #   规格 llm:<api_base>|<model>|<api_key>   任一段可空，空段回退全局环境变量
  python scripts/self_play.py \
      --black "llm:https://openrouter.ai/api/v1|stealth/ox-alpha|KEY1" \
      --white "llm:https://api.openai.com/v1|gpt-4o|KEY2"

  # 也兼容旧写法：仅给 model，端点走环境变量
  #   export LLM_API_BASE=...   LLM_API_KEY=sk-...   LLM_MODEL=ox-alpha

参数说明：
  --black / --white : 棋手规格，格式为 "类型[:参数]"
      heuristic:<防守权重>             启发式基线
      console                          控制台人工/代理输入
      llm:<模型名>                     单 LLM 自我对弈（端点走环境变量）
      llm:<api_base>|<model>|<api_key> 任意两个 LLM 互搏，端点内联
  --best-of              系列赛局数（奇数，默认 3）
  --out                  系列赛 JSON 日志输出路径（默认 games/series.json）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 让脚本能直接 import gomoku 包（无论以何种路径运行）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gomoku.board import Stone  # noqa: E402
from gomoku.heuristics import HeuristicPlayer  # noqa: E402
from gomoku.players import ConsolePlayer, LLMPlayer  # noqa: E402
from gomoku.referee import Referee  # noqa: E402
from gomoku.search import SearchPlayer  # noqa: E402


def build_player(spec: str, name: str, color: Stone):
    """按规格字符串构造棋手并设置执子颜色。"""
    kind, _, arg = spec.partition(":")
    if kind == "heuristic":
        dw = float(arg) if arg else 1.0
        p = HeuristicPlayer(name, defense_weight=dw)
    elif kind == "search":
        depth = int(arg) if arg else 4
        p = SearchPlayer(name, depth=depth)
    elif kind == "console":
        p = ConsolePlayer(name)
    elif kind == "llm":
        # 支持两种写法：
        #   1) llm:<模型名>                                —— 端点走全局环境变量
        #   2) llm:<api_base>|<model>|<api_key>             —— 任意两段可空，空段回退环境变量
        # 这样即可让"任意两个 LLM（含跨服务商）"互搏，或"单个 LLM 自我对弈"。
        if "|" in arg:
            parts = arg.split("|")
            api_base = (parts[0].strip() if len(parts) > 0 and parts[0].strip() else "") or os.environ.get("LLM_API_BASE", "")
            model = (parts[1].strip() if len(parts) > 1 and parts[1].strip() else "") or os.environ.get("LLM_MODEL", "")
            # key 段可空：本地 vLLM/Ollama 等无鉴权端点允许省略（不用或链回退，避免空串拦截）
            api_key = (parts[2].strip() if len(parts) > 2 and parts[2].strip() else "") or os.environ.get("LLM_API_KEY", "")
        else:
            model = arg.strip() or os.environ.get("LLM_MODEL", "")
            api_base = os.environ.get("LLM_API_BASE", "")
            api_key = os.environ.get("LLM_API_KEY", "")
        if not (model and api_base):
            raise SystemExit(
                "使用 llm 棋手需提供 model（规格内或 LLM_MODEL），"
                "以及 LLM_API_BASE（规格内或环境变量）；api_key 仅鉴权端点需要"
            )
        p = LLMPlayer(name, model, api_base, api_key)
    else:
        raise SystemExit(f"未知棋手类型: {kind}")
    p.set_stone(color)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="五子棋 LLM 对战运行器（三局两胜）")
    parser.add_argument("--black", default="heuristic:1.0", help="黑方棋手规格")
    parser.add_argument("--white", default="heuristic:1.2", help="白方棋手规格")
    parser.add_argument("--black-name", default=None, help="黑方显示名（默认使用规格字符串）")
    parser.add_argument("--white-name", default=None, help="白方显示名（默认使用规格字符串）")
    parser.add_argument("--best-of", type=int, default=3, help="系列赛局数（奇数）")
    parser.add_argument("--out", default=os.path.join(ROOT, "games", "series.json"),
                        help="系列赛 JSON 日志输出路径")
    parser.add_argument("--live-log", default=None,
                        help="逐手实时落盘路径（JSONL，默认 <out 同目录>/live_log.jsonl；"
                             "给 off 关闭）")
    args = parser.parse_args()

    black = build_player(args.black, name=args.black_name or args.black, color=Stone.BLACK)
    white = build_player(args.white, name=args.white_name or args.white, color=Stone.WHITE)

    # 实时落盘默认与 series.json 同目录，便于观战与崩溃挽留
    live_log = args.live_log
    if live_log is None:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        live_log = os.path.join(out_dir, "live_log.jsonl")
    if live_log.lower() == "off":
        live_log = None

    referee = Referee(black, white, live_log_path=live_log)
    try:
        series = referee.play_series(best_of=args.best_of)
    except Exception as exc:
        # 极端情况下系列赛中断：仍把已累积的原始返回落盘，保证可复盘
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        for tag, p in (("black", black), ("white", white)):
            raw = getattr(p, "raw_log", None)
            if raw:
                with open(os.path.join(out_dir, f"{tag}_raw.json"), "w", encoding="utf-8") as f:
                    json.dump(raw, f, ensure_ascii=False, indent=2)
                print(f"  [异常兜底] {tag} 原始返回日志已保存（中断前共 {len(raw)} 手）")
        print(f"[系列赛中断] {exc}")
        raise

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    referee.save_series_log(series, args.out)

    # 收集 LLM 棋手每手原始返回（content/reasoning/解析结果），供赛后复盘
    out_dir = os.path.dirname(os.path.abspath(args.out))
    for tag, p in (("black", black), ("white", white)):
        raw = getattr(p, "raw_log", None)
        if raw:
            raw_path = os.path.join(out_dir, f"{tag}_raw.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
            print(f"  {tag} 原始返回日志（可复盘）：{raw_path}")

    # 控制台摘要
    print("=" * 60)
    print(f"系列赛结果（三局两胜，best_of={series.best_of}）")
    print(f"  黑方（第1局）{series.games[0].black_name}")
    print(f"  白方（第1局）{series.games[0].white_name}")
    # series.wins 已按"选手名"累计（修复前曾误按座位色），按第1局黑白顺序取数
    b_name = series.games[0].black_name
    w_name = series.games[0].white_name
    print(f"  比分 {b_name} {series.wins.get(b_name, 0)} : {series.wins.get(w_name, 0)} {w_name}")
    for g in series.games:
        print(f"  第 {g.game_index + 1} 局 -> 胜方 {g.winner} ({g.winner_name})，"
              f"原因 {g.reason}，手数 {len(g.moves)}，用时 {g.duration_ms}ms")
    print(f"  系列赛冠军：{series.series_winner_name}")
    print(f"  日志已保存：{args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
