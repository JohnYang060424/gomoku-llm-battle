"""棋手（运动员）抽象与具体实现。

本模块定义统一棋手接口 Player，并内置两种具体棋手：
- ConsolePlayer：通过标准输入落子，用于本地人工对弈 / 调试；
- LLMPlayer    ：调用 OpenAI 兼容接口，由大模型决策落子，
                 这是"LLM 之间五子棋对战"的核心运动员。

注意：HeuristicPlayer（启发式基线）定义在 heuristics.py，
避免本文件与启发式模块产生循环依赖。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from . import config
from .board import Board, Move, Stone
from . import llm


class Player(ABC):
    """棋手抽象基类。

    属性：
        name : 棋手名称（用于日志 / 显示）
        stone: 本局执子颜色（由裁判在开局前 set_stone 设定）
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.stone: Optional[Stone] = None
        self.hint: Optional[str] = None  # 裁判注入的对手威胁提示（LLM 棋手拼进提示词）

    def set_stone(self, stone: Stone) -> None:
        self.stone = stone

    @abstractmethod
    def choose_move(self, board: Board) -> Move:
        """根据当前棋盘返回下一步落子。子类必须实现。"""
        raise NotImplementedError


class ConsolePlayer(Player):
    """控制台棋手：把棋盘打印出来，从标准输入读取落子。

    适用于人工参与或代理输入的演示场景。
    input_func 可注入，便于脚本化测试。
    """

    def __init__(self, name: str = "Human", input_func: Callable[[str], str] = input) -> None:
        super().__init__(name)
        self._input = input_func

    def choose_move(self, board: Board) -> Move:
        print(board.render())
        print(f"轮到 {self.name}（{self.stone.name}）落子，请输入坐标，形如 7,7：")
        while True:
            raw = self._input("> ").strip()
            move = _parse_move_text(raw, board)
            if move is not None:
                return move
            print("无法解析该坐标，请重新输入（0 索引，row,col）：")


class LLMPlayer(Player):
    """大模型棋手：把棋盘状态翻译成提示词，调用 LLM 决策落子。

    任意 OpenAI 兼容端点均可接入，任意两个 LLM 都可用同一套代码驱动，
    只需传入不同 api_base / model 即可。
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_base: str,
        api_key: str,
        *,
        temperature: float = config.LLM_TEMPERATURE,
        timeout: int = config.LLM_TIMEOUT,
        max_retries: int = config.LLM_MAX_RETRIES,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> None:
        super().__init__(name)
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.raw_log: list = []  # 每手原始返回（content/reasoning/解析结果），供复盘

    def choose_move(self, board: Board) -> Move:
        messages = self._build_messages(board)
        turn = len(board.moves) + 1
        last_text = ""
        for attempt in range(self.max_retries):
            try:
                text, message = llm.call_chat_completion(
                    self.api_base,
                    self.api_key,
                    self.model,
                    messages,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    max_tokens=self.max_tokens,
                    return_raw=True,
                )
            except Exception as exc:  # 网络/接口异常：重试
                last_text = f"(请求异常: {exc})"
                self.raw_log.append({
                    "turn": turn,
                    "attempt": attempt,
                    "color": self.stone.name if self.stone else None,
                    "error": str(exc),
                    "parsed": None,
                    "legal": False,
                })
                messages = self._append_retry(messages, last_text)
                continue

            move = _parse_move_text(text, board)
            self.raw_log.append({
                "turn": turn,
                "attempt": attempt,
                "color": self.stone.name if self.stone else None,
                "content": message.get("content"),
                "reasoning": message.get("reasoning") or message.get("reasoning_content"),
                "parsed": str(move) if move else None,
                "legal": move is not None,
            })
            if move is not None:
                return move
            # 解析失败：把模型的原始输出反馈回去，要求重答
            last_text = text
            messages = self._append_retry(messages, f"非法或越界落子：{text!r}")
        # 重试耗尽仍失败：降级为保底合法落子（启发式贪心），保证对局不中断、可复盘。
        # 这样即便 LLM 持续返回非法/被截断，系列赛也能跑完并保留完整日志。
        fallback = self._fallback_move(board)
        self.raw_log.append({
            "turn": turn,
            "color": self.stone.name if self.stone else None,
            "error": f"重试 {self.max_retries} 次耗尽，降级保底落子",
            "parsed": str(fallback),
            "legal": True,
            "fallback": True,
        })
        print(f"  [LLM {self.name} FALLBACK] 重试耗尽，改用保底落子 {fallback}")
        return fallback

    def _fallback_move(self, board: Board) -> Move:
        """保底落子：启发式评估全盘候选，取综合分最高者；无候选则取首个空点。

        仅在 LLM 重试全部失败（持续非法/响应截断）时启用，确保对局不中断。
        """
        # 延迟导入，避免与 heuristics 模块形成顶层循环依赖
        from .heuristics import evaluate_move, generate_candidates
        cands = generate_candidates(board)  # List[(r, c)]
        if cands:
            r, c = max(cands, key=lambda rc: sum(evaluate_move(board, rc[0], rc[1], self.stone)))
            return Move(r, c)
        for r in range(board.size):
            for c in range(board.size):
                if board.is_empty(r, c):
                    return Move(r, c)
        raise ValueError("棋盘已满，无法生成保底落子")

    # ------------------------- 提示词构造 -------------------------
    def _build_messages(self, board: Board) -> List[dict]:
        stone_name = "黑棋(BLACK, 用 X 表示)" if self.stone is Stone.BLACK else "白棋(WHITE, 用 O 表示)"
        system = (
            "你是一名五子棋（Gomoku）高手。规则：15x15 棋盘，任意方向（横/竖/两条斜线）"
            "先连成 5 子者获胜（长连也胜）。你只能落在空点（.）。"
            "请基于棋理（进攻成五、封堵对手活四/冲四、抢活三）选择最优落子。"
            "只输出一个坐标，格式严格为 'row,col'（0 索引，row 在上、col 在右），不要任何解释。"
        )
        user = (
            f"你是 {stone_name}。\n"
            f"当前棋盘（左侧数字为 row，顶部数字为 col；. 空 X 黑 O 白）：\n\n"
            f"{board.render()}\n\n"
        )
        # 机械裁判扫描到的对手威胁告警（象棋"将军!"的对等物）：
        # LLM 常"看得到"对手连子却不封堵，由裁判明确指出最高危威胁并给出
        # 建议封堵点，强制其应对（实测：无告警时落边角被速杀，带告警及时封堵）。
        if self.hint:
            user += (
                f"⚠️ 裁判强制告警：{self.hint}\n"
                f"必须把落子下在告警给出的封堵点上（若多个选一个），"
                f"否则对手下一手即成五。不得先走自己的进攻！\n\n"
            )
        user += "请输出你的落子坐标（row,col）："
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _append_retry(messages: List[dict], feedback: str) -> List[dict]:
        """把模型上一轮的错误输出作为 assistant 消息，并追加纠错 user 消息。"""
        messages = list(messages)
        messages.append({"role": "assistant", "content": feedback})
        messages.append({
            "role": "user",
            "content": "该坐标非法或越界，请重新输出一个空点坐标，格式严格为 'row,col'。",
        })
        return messages


# ------------------------- 坐标解析工具 -------------------------
# 兼容半角/全角括号、空格变体："7,7" "(7,7)" "（7，7）" "7，7"
_MOVE_RE = re.compile(r"[（(]?\s*(\d{1,2})\s*[,，]\s*(\d{1,2})\s*[)）]?")


def _parse_move_text(text: str, board: Board) -> Optional[Move]:
    """从模型/用户输入文本中解析出合法的 'row,col' 落子。

    策略：倒序遍历所有匹配——推理模型常在思考过程中提及多个坐标，
    最终决策通常在响应末尾。取最后一个落在棋盘范围内的空点坐标。
    兼容 "r,c" 与 "(r,c)" 两种写法。
    """
    if not text:
        return None
    matches = list(_MOVE_RE.finditer(text))
    if not matches:
        return None
    # 倒序检查：优先取响应末尾的坐标（更贴近最终决策）
    for match in reversed(matches):
        a = match.group(1) if match.group(1) is not None else match.group(3)
        b = match.group(2) if match.group(2) is not None else match.group(4)
        r, c = int(a), int(b)
        if 0 <= r < board.size and 0 <= c < board.size and board.is_empty(r, c):
            return Move(r, c)
    return None
