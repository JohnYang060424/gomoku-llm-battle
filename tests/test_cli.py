#!/usr/bin/env python3
"""CLI 棋手规格解析测试（不联网）。

验证 self_play.build_player 对以下情况的正确性：
  - 单 LLM 自我对弈：llm:<model>（端点走环境变量）
  - 任意两个 LLM 互搏：llm:<api_base>|<model>|<api_key>（含跨服务商）
  - 环境变量回退与缺失报错的边界处理
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.self_play import build_player  # noqa: E402
from gomoku.players import LLMPlayer  # noqa: E402
from gomoku.board import Stone  # noqa: E402


class TestCLILLMParsing(unittest.TestCase):
    """验证 llm 规格的多种写法被正确解析为独立端点。"""

    def test_inline_full_spec(self):
        """内联三段：api_base|model|api_key 全部解析正确。"""
        p = build_player(
            "llm:https://openrouter.ai/api/v1|stealth/ox-alpha|sk-abc",
            "B",
            Stone.BLACK,
        )
        self.assertIsInstance(p, LLMPlayer)
        self.assertEqual(p.model, "stealth/ox-alpha")
        self.assertEqual(p.api_base, "https://openrouter.ai/api/v1")
        self.assertEqual(p.api_key, "sk-abc")
        self.assertEqual(p.stone, Stone.BLACK)

    def test_cross_provider_independent(self):
        """跨服务商：两方端点互相独立，可同时驱动。"""
        b = build_player(
            "llm:https://openrouter.ai/api/v1|stealth/ox-alpha|KEY_A",
            "B",
            Stone.BLACK,
        )
        w = build_player(
            "llm:https://api.openai.com/v1|gpt-4o|KEY_B",
            "W",
            Stone.WHITE,
        )
        self.assertEqual(b.api_base, "https://openrouter.ai/api/v1")
        self.assertEqual(w.api_base, "https://api.openai.com/v1")
        self.assertNotEqual(b.model, w.model)
        self.assertNotEqual(b.api_key, w.api_key)

    def test_single_model_self_play(self):
        """单个 LLM 自我对弈：llm:<model> 简写，端点走环境变量。"""
        os.environ["LLM_API_BASE"] = "https://env.base/v1"
        os.environ["LLM_API_KEY"] = "sk-env"
        try:
            b = build_player("llm:gpt-4o", "B", Stone.BLACK)
            w = build_player("llm:gpt-4o", "W", Stone.WHITE)
            self.assertIsInstance(b, LLMPlayer)
            self.assertIsInstance(w, LLMPlayer)
            self.assertEqual(b.model, w.model)
            self.assertEqual(b.api_base, w.api_base)
            self.assertEqual(b.stone, Stone.BLACK)
            self.assertEqual(w.stone, Stone.WHITE)
        finally:
            del os.environ["LLM_API_BASE"]
            del os.environ["LLM_API_KEY"]

    def test_env_fallback(self):
        """规格仅给 model，api_base/api_key 回退到环境变量。"""
        os.environ["LLM_API_BASE"] = "https://env.base/v1"
        os.environ["LLM_API_KEY"] = "sk-env"
        try:
            p = build_player("llm:gpt-4o", "B", Stone.BLACK)
            self.assertEqual(p.model, "gpt-4o")
            self.assertEqual(p.api_base, "https://env.base/v1")
            self.assertEqual(p.api_key, "sk-env")
        finally:
            del os.environ["LLM_API_BASE"]
            del os.environ["LLM_API_KEY"]

    def test_partial_inline_falls_back_to_env(self):
        """内联只给 model，其余两段空，回退环境变量。"""
        os.environ["LLM_API_BASE"] = "https://env2.base/v1"
        os.environ["LLM_API_KEY"] = "sk-env2"
        try:
            # 空 api_base 与空 api_key（|| 写法）
            p = build_player("llm:|gpt-4o|", "B", Stone.BLACK)
            self.assertEqual(p.model, "gpt-4o")
            self.assertEqual(p.api_base, "https://env2.base/v1")
            self.assertEqual(p.api_key, "sk-env2")
        finally:
            del os.environ["LLM_API_BASE"]
            del os.environ["LLM_API_KEY"]

    def test_missing_config_raises(self):
        """既无内联也无环境变量时，抛出 SystemExit 提示。"""
        saved = {k: os.environ.pop(k, None) for k in
                 ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")}
        try:
            with self.assertRaises(SystemExit):
                build_player("llm:gpt-4o", "B", Stone.BLACK)
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
