"""LLM 接口封装（OpenAI 兼容 /chat/completions）。

设计目标：
- 零第三方依赖，仅用标准库 urllib 发起 HTTPS 请求；
- 适配任意 OpenAI 兼容端点（OpenAI / OpenRouter / 自建 vLLM / ox-alpha 等）；
- 调用失败可重试，返回结构化的文本响应，由上层解析为落子坐标。
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

from . import config


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = config.LLM_TEMPERATURE,
    timeout: int = config.LLM_TIMEOUT,
    max_tokens: int = config.LLM_MAX_TOKENS,
    return_raw: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """调用一次聊天补全。

    默认返回助手消息中提取的纯文本（已兼容 reasoning 字段的推理模型）；
    当 return_raw=True 时返回 (text, message_dict) 元组，message_dict 含
    content / reasoning 等原始字段，供上层做复盘日志。
    max_tokens 限制响应长度，防止推理模型 thinking 过长导致超时。
    """
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # vLLM 后端严格执行 enable_thinking：不显式传则按服务端默认，
    # 显式开启保证 27B 等推理模型带思考链下棋（棋力显著更强）。
    if getattr(config, "LLM_ENABLE_THINKING", True):
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    try:
        body: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        # 响应可能不完整：推理模型（如 ox-alpha）的超长 reasoning 会触发
        # 服务端响应长度上限，导致 JSON 在中间被截断。此时尝试从残缺文本中
        # 抢救最后一个合法坐标（响应体只含模型自身输出，不含棋盘渲染数字，
        # 其中的 'r,c' 必为模型意图落子）。
        rescued, message = _rescue_truncated(raw)
        if rescued is None:
            raise ValueError(
                f"LLM 响应 JSON 解析失败且无法从截断内容抢救坐标（原始长度 {len(raw)}）"
            )
        if return_raw:
            return rescued, message
        return rescued
    message = body["choices"][0]["message"]
    text = _extract_move_text(message)
    if return_raw:
        return text, message
    return text


# 用于从截断响应中抢救坐标：匹配 'r,c' / '(r,c)' 形式（r,c ∈ 0..14）
_RESCUE_RE = re.compile(r"(\d{1,2})\s*,\s*(\d{1,2})")


def _rescue_truncated(raw: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """从被截断的 LLM 响应文本中抢救坐标。

    响应体仅包含模型自身的 content/reasoning（不含棋盘渲染数字），
    因此其中的 'r,c' 均为模型意图落子。取最后一个落在棋盘范围内
    (0..14) 的坐标为决策，更贴近其最终输出。
    """
    for r_s, c_s in reversed(_RESCUE_RE.findall(raw)):
        r, c = int(r_s), int(c_s)
        if 0 <= r <= 14 and 0 <= c <= 14:
            text = f"{r},{c}"
            return text, {"content": text, "reasoning": "(从截断响应抢救)"}
    return None, {}


def _extract_move_text(message: Dict[str, Any]) -> Optional[str]:
    """从助手消息中提取可作为落子的文本。

    ox-alpha 等推理模型常把坐标放在 reasoning 字段、content 为 null，
    因此优先取 content，为空时回退到 reasoning 中【末尾】的 'r,c' 形式坐标
    （取末尾更贴近其最终决策，而非中间讨论过程）。
    """
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if isinstance(reasoning, str):
        matches = re.findall(r"(\d+)\s*,\s*(\d+)", reasoning)
        if matches:
            r, c = matches[-1]
            return f"{r},{c}"
    return content  # 可能为 None，上层解析会得到 None -> 触发重试
