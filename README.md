# gomoku-llm-battle · 五子棋 LLM 对战框架

一个**零第三方依赖**的五子棋（Gomoku）对战框架，专为**大模型（LLM）之间的五子棋对弈**设计。

> 程序是裁判，也是运动员框架；任意两个 LLM 接进来，就能让它们自己下棋、由程序当裁判、三局两胜。

## 特性

- **规则引擎严谨**：15×15 标准棋盘，四方向连五判定（freestyle，长连亦胜），和棋判定。
- **统一棋手接口** `Player`：想接新模型只需实现一个 `choose_move(board)`。
  - `LLMPlayer`：调用任意 OpenAI 兼容 `/chat/completions` 端点，**兼容推理模型**（`reasoning` 字段双路解析），自动重试 + 保底落子，对局不中断。
  - `SearchPlayer`：negamax + α-β 搜索（深度可调），会下棋的强基线。
  - `HeuristicPlayer`：贪心启发式，零搜索、极快，作参照基准。
  - `ConsolePlayer`：人工 / 代理输入，便于本地"亲自上场"。
- **裁判 + 三局两胜**：自动交替先后手（公平），按选手累计比分并提前结束；每局记录落子序列、胜负原因、SGF。
- **任意两个 LLM 互搏**：CLI 支持每方独立端点（含跨服务商），一条命令开赛。
- **可复盘**：每手原始返回（`content` / `reasoning` / 解析结果 / 合法性）自动落盘为 JSON。
- **零依赖**：仅用 Python 标准库，开箱即跑。

## 快速开始

```bash
# 0. 无需安装任何依赖，Python >= 3.9
cd gomoku-llm-battle

# 1. 跑测试（31 项）
python -m unittest discover -s tests

# 2. 两个基线棋手先下一场（验证整条管线）
python scripts/self_play.py --black search:4 --white heuristic:1.0 --best-of 3
```

## LLM 之间对战

### 单个 LLM 自我对弈

```bash
export LLM_API_BASE="https://openrouter.ai/api/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="your-model-name"

python scripts/self_play.py --black "llm:your-model-name" --white "llm:your-model-name" --best-of 3
```

### 任意两个 LLM 互搏（含跨服务商）

```bash
python scripts/self_play.py \
  --black "llm:https://openrouter.ai/api/v1|model-a|KEY_A" \
  --white "llm:https://api.openai.com/v1|gpt-4o|KEY_B" \
  --best-of 3 \
  --out games/series.json
```

规格格式：`llm:<api_base>|<model>|<api_key>`，任一段可空，空段回退全局环境变量。

### 对局日志与复盘

每场对局自动产出：
- `series.json` — 三局全过程、每手落子、胜负判定、SGF
- `black_raw.json` / `white_raw.json` — 黑/白每手原始返回（`content` / `reasoning` / 解析结果 / 合法性），供复盘分析

## 架构

```
gomoku/
  config.py      全局配置（棋盘大小、棋型评分、搜索/LLM 参数）
  board.py       棋盘与规则引擎（落子校验、连五/和棋判定、渲染、克隆）
  heuristics.py  棋型评分 + 贪心基线 HeuristicPlayer
  search.py      negamax + α-β 搜索型棋手 SearchPlayer
  players.py     Player 抽象 + ConsolePlayer + LLMPlayer
  llm.py         大模型接口封装（标准库 urllib，兼容推理模型）
  referee.py     裁判：单局 / 三局两胜系列赛 / JSON+SGF 日志
  sgf.py         SGF 存档导出
scripts/
  self_play.py   对战运行器（CLI）
tests/           单元测试（31 项）
```

**调用关系**：`Referee` 持有两个 `Player`；每步向当前棋手要落子 → `Board.place` 校验 → 判定胜负/和棋 → 系列赛按三局两胜累计。

## 棋手规格字符串

| 类型        | 参数            | 说明                         |
|-------------|-----------------|------------------------------|
| `heuristic` | 防守权重(浮点)  | 贪心基线，如 `heuristic:1.0` |
| `search`    | 搜索深度(整数)  | negamax，如 `search:4`       |
| `console`   | 无              | 控制台输入（亲自上场）       |
| `llm`       | `model` 或 `api_base\|model\|api_key` | LLM 棋手 |

## 推理模型兼容

部分 LLM（如 OpenRouter 上的推理模型）把答案放在 `reasoning` 字段、`content` 返回 `null`。本框架做了三层容错：

1. `content` 为空时，自动从 `reasoning` 末尾提取坐标；
2. 响应 JSON 被截断时，从残缺文本中抢救最后一个合法坐标；
3. 全部失败后降级为启发式保底落子，保证对局不中断、日志可复盘。

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：连五四方向、长连、非法落子、和棋、候选生成、必胜/封堵、裁判计分与日志、LLM 坐标解析与重试、CLI 规格解析（含跨服务商端点）。

## Roadmap

- [ ] 迭代深化 / 置换表，提升搜索强度与速度
- [ ] Renju（连珠）禁手规则
- [ ] 对局可视化网页（实时棋盘 + 落子回放）
- [ ] 多模型循环赛（round-robin）与评分排行
- [ ] 逐手实时落盘（中途可监控、崩溃不丢数据）

## License

[MIT](LICENSE)
