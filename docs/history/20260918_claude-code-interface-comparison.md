---
date: 2026-09-18
kind: report
---

# Claude Code / Claude Agent SDK のインターフェースと Kennel が備えるべきもの

作成日: 2026-09-18 検証元: `claude` CLI 2.1.274 の `--help` 出力、`claude-agent-sdk` (Python) 0.2.154 の実 export、および公式ドキュメント。 Kennel 側は v0.0.3 のコード (`src/kennel/`) から抽出。

凡例(「Kennel」列):

| 記号 | 意味 |
| --- | --- |
| ✅ | v0.0.3 に既にある(✅ A は 2026-09-18 に着地した優先度 A 項目、main `329ef46`) |
| 🔜 | 備えるべき(追加候補、優先度 A/B/C) |
| 🔁 | 形を変えて備えるべき(Kennel の設計に合わせて縮小・置換) |
| ❌ | 対象外(spec の非目標、またはオンデバイス前提に合わない) |

判断の基準は spec の設計原則: **SDK が本体で CLI は consumer**、**local-first**、**workspace boundary が中核の安全不変条件**、**オンデバイス 3B モデル(コンテキスト約 4k)前提**、**v0.1 非目標**(Claude Code 完全互換・IDE/GUI・マルチユーザー・MCP 完全実装・subagent・OS サンドボックス・常駐 daemon)。

> **更新(2026-09-18)**: 優先度 A の 10 項目は Issue #1〜#10 → PR #11〜#20 として実装され、Land PR #27(main `329ef46`)で着地した。表中の「✅ A」がそれに当たる。ただし #19 は「ターン単位の usage 推定」を撤回したため `AgentResult.usage` はプロバイダ実測があるときだけ埋まる(Apple では `None`)。統合時の後片付けは Issue #21〜#26。

---

## 1. Claude Code CLI (`claude`) のインターフェース一覧

### 1.1 サブコマンド

| サブコマンド | 役割 | Kennel |
| --- | --- | --- |
| `claude [prompt]` | 対話セッション開始 | ✅ `kennel [workspace]` |
| `claude -p` | 非対話(print)モード | ✅ `-p/--prompt` |
| `agents` / `attach` / `logs` / `stop`,`kill` / `rm` / `respawn` | バックグラウンドセッション管理 | ❌ daemon は非目標 |
| `auth login/logout/status`, `setup-token` | 認証 | ❌ 認証不要(オンデバイス) |
| `mcp` | MCP サーバー設定 | 🔁 C: 将来 MCP bridge を入れるなら `kennel mcp add/list` |
| `plugin` | プラグイン管理 | 🔁 C: entry-point 方式のツールプラグイン発見のみ |
| `doctor` | インストール診断 | ✅ A: `kennel doctor`(Xcode/Apple Intelligence/モデル可用性/設定ファイル検証。現在は `-p hello` の exit 3 で代用している) |
| `update`,`upgrade` / `install` | 自己更新 | ❌ pip/uv に任せる |
| `project purge` | プロジェクト状態管理 | 🔜 B: 永続セッション導入後に `kennel sessions list/rm` |
| `auto-mode`, `gateway`, `import`, `ultrareview` | 企業向け/クラウド機能 | ❌ |

### 1.2 主要フラグ

| フラグ | 役割 | Kennel |
| --- | --- | --- |
| `-p, --print` | 一発実行 | ✅ `-p TEXT`(Claude Code はプロンプトを位置引数、Kennel はフラグ値。Kennel の位置引数は workspace) |
| `--output-format text/json/stream-json` | 出力形式 | ✅ A: `json` = `AgentResult` を 1 オブジェクトで、`stream-json` = 既存イベントの JSONL を stdout に。現在 `--trace` が stderr に同等物を出す |
| `--input-format stream-json` | 標準入力から連続入力 | 🔜 C |
| `--json-schema` | 構造化出力 | ✅ A: SDK に `respond_structured` が既にあるので CLI へ露出するだけ |
| `-c, --continue` / `-r, --resume` / `--session-id` / `--fork-session` / `-n, --name` | セッション継続 | 🔜 B: 永続セッション(spec §29 は「設計上阻害しない」) |
| `--no-session-persistence` | 保存しない | 🔜 B(永続化と同時) |
| `--permission-mode default/acceptEdits/plan/dontAsk/bypassPermissions/auto` | 権限モード | 🔁 A: `--permission-mode read-only/default/accept-edits/dont-ask/bypass` を導入し、既存 `--read-only`/`--allow-write`/`--non-interactive` はその糖衣に。`auto`(分類器)は ❌ |
| `--allowedTools` / `--disallowedTools` / `--tools` | ツール許可・選択 | ✅ A(一部): ルール構文 `shell(git *)` `write(docs/**)` は `kennel.json` / `Agent(permissions=)` で着地(PR #17)。CLI の `--tools` / `--allow` / `--deny` フラグは未着地(🔜 B) |
| `--dangerously-skip-permissions` | 全許可 | 🔁 A: `--permission-mode bypass`(要確認プロンプト付き。shell を含む) |
| `--restricted` / `--safe-mode` | 縮小起動 | 🔁 B: `--read-only` が `--restricted` 相当。`--safe-mode` は設定ファイルを無視する `--no-config` として |
| `--system-prompt` / `--append-system-prompt` (+ `-file`) | 指示差し替え/追記 | ✅ A: `--instructions TEXT|@file`(追記)、`--system-prompt`(差し替え)。設定の `agent.instructions` は既にある |
| `--model` / `--fallback-model` / `--effort` / `--fast` / `--max-thinking-tokens` | モデル選択 | 🔁 B: `--provider`(現在 hidden)を公開し `--model` の別名に。effort/thinking は Apple FM に無いので ❌ |
| `--max-turns` / `--max-budget-usd` | 停止条件 | 🔁: コスト概念が無いので `--max-tool-calls`(✅)と `--turn-timeout`(🔜 B, 設定には既にある) |
| `--add-dir` | 追加ディレクトリ | ❌ v0.1(単一 workspace boundary が不変条件)。将来は複数 root を `Workspace` が扱えるようになってから |
| `--mcp-config` / `--strict-mcp-config` | MCP | 🔜 C(MCP bridge と同時) |
| `--settings FILE|JSON` / `--setting-sources` | 設定注入 | 🔜 B: `--config FILE|JSON`、`--no-user-config` |
| `--agents` / `--agent` | サブエージェント | ❌ 非目標 |
| `--permission-prompt-tool` / `--permission-prompts host|none` | 権限応答者 | ✅ `--non-interactive` = `none`。`host` は SDK の `prompter` |
| `--include-partial-messages` / `--include-hook-events` / `--replay-user-messages` / `--forward-subagent-text` | stream-json 詳細 | 🔁 B: `model.delta` イベントを stream-json に含めるか否かのフラグ 1 つで足りる |
| `--verbose` / `-d, --debug` / `--debug-file` | 診断 | ✅ `--verbose`, `--trace`。🔜 C: `--trace-file PATH` |
| `--autocompact` | 自動コンパクション閾値 | 🔁 B: Kennel は `ContextLimitError` で自動コンパクションする(✅)。閾値指定は 4k 窓では無意味なので、`--no-compact` のみ |
| `--bare` | フック等を全部切る | 🔁 C: `--no-config` に含める |
| `-w, --worktree`, `--tmux`, `--bg`, `--cloud`, `--remote-control`, `--teleport`, `--environment`, `--chrome`, `--ide`, `--betas`, `--file`, `--plugin-dir/url`, `--prompt-suggestions`, `--brief`, `--ax-screen-reader`, `--exclude-dynamic-system-prompt-sections`, `--system-prompt-snapshot` | クラウド/IDE/git/UI 系 | ❌ |
| `-v, --version` / `-h, --help` | | ✅ |

### 1.3 対話スラッシュコマンド

| コマンド | Kennel |
| --- | --- |
| `/help`, `/clear`, `/exit` | ✅ |
| `/status` | ✅(session, model, tools, permissions) |
| `/compact` | ✅ A: `Session._compact()` が既にあるので公開するだけ |
| `/permissions` | ✅ A: 現在のポリシー表示・`/permissions shell allow` で変更(`PermissionManager.set_decision` は既にある) |
| `/tools` | 🔜 B: 有効ツールと制限値の表示(Claude Code は `/status` と `/permissions` に含む) |
| `/usage`, `/cost`, `/context` | 🔁 A: `/usage` でトークン数・ターン数・コンパクション回数・**コンテキスト残量**。オンデバイスは費用ゼロなので `/cost` は不要 |
| `/resume`, `/fork`, `/branch`, `/export` | 🔜 B(永続セッションと同時。`/export` はまず transcript を Markdown 保存) |
| `/config`, `/model` | 🔜 C: `/config` は有効設定と出典(`KennelConfig.sources` が既にある)の表示。`/model` は provider が複数になってから |
| `/init`, `/memory` | 🔜 B: `KENNEL.md`(spec §38)を作る/開く |
| `/plan` | 🔁: `/permissions read-only` で代替 |
| `/doctor`, `/mcp`, `/agents`, `/hooks`, `/rewind` | 🔜 C(それぞれの機能導入後) |
| `/code-review`, `/diff`, `/security-review`, `/deep-research`, `/loop`, `/goal`, `/batch`, `/design*`, `/dataviz`, `/tasks`, `/subtask`, `/list-agents`, `/teleport`, `/remote-control`, `/login`, `/logout`, `/mobile`, `/desktop`, `/upgrade`, `/passes`, `/bug`, `/feedback`, `/copy`, `/btw`, `/color`, `/theme`, `/keybindings`, `/vim`, `/ide`, `/install-github-app`, `/import`, `/cd`, `/add-dir`, `/background`, `/effort`, `/fast`, `/advisor`, `/autofix-pr` | ❌ |

### 1.4 組み込みツール

| Claude Code | Kennel |
| --- | --- |
| Read, Write, Edit, Glob, Grep | ✅ read, write, edit, glob, grep |
| Bash (PowerShell, Monitor) | ✅ shell(別権限)。Monitor ❌ |
| WebSearch, WebFetch | ✅ web(既定 off、検索プロバイダ要設定)。🔜 C: `web_fetch` を分離 |
| AskUserQuestion | 🔜 B: `ask_user` ツール。小型モデルは「どのファイル?」と聞き返しがちで、narration guard で吸収している問題の正攻法。SDK では prompter と同型の callback |
| Agent(subagent), TaskCreate/List/Get/Update/Stop, CronCreate/List/Delete, ScheduleWakeup, SendMessage, ListAgents | ❌ 非目標 |
| NotebookEdit, LSP | ❌ |
| EnterPlanMode/ExitPlanMode, EnterWorktree/ExitWorktree, Skill, Workflow, Artifact, PushNotification, SendUserFile, RemoteTrigger, ReportFindings, SendFeedback, EndConversation, ListMcpResourcesTool/ReadMcpResourceTool | ❌ |
| (MCP サーバーのツール) | 🔜 C: MCP stdio サーバーを Kennel `Tool` として橋渡し(spec §38「MCP bridge」) |

### 1.5 設定面

| Claude Code | Kennel |
| --- | --- |
| `~/.claude/settings.json` → `.claude/settings.json` → `.claude/settings.local.json` → managed | ✅ `~/.config/kennel/settings.json` → `./kennel.json`(precedence: CLI > `Agent()` > project > user > defaults)。🔜 B: `kennel.local.json`(gitignore 前提、承認済みルールの書き戻し先) |
| `permissions.allow/ask/deny` のルール構文 `Tool(specifier)` | ✅ A: `"permissions": {"shell": "ask", "shell(git *)": "allow", "write(docs/**)": "allow"}`。ツール種別単位(✅)を基底にしたまま specifier を追加 |
| `CLAUDE.md`(user/project/local の階層) | 🔜 B: `KENNEL.md`(project)+ `~/.config/kennel/KENNEL.md`(user)。3B モデルなので短さの制約を README で明示 |
| `.claude/commands`, `agents`, `skills`, `rules`, plugins/marketplace | ❌(commands は 🔜 C: `.kennel/commands/*.md` のプロンプトテンプレート程度) |
| `hooks` 設定 | 🔁 B(§1.6) |
| `env`, `model`, `sandbox`, `outputStyle`, `statusLine` | `env` ✅(`Agent(environment=)`, 🔜 C で設定ファイルにも)。他 ❌ |
| 環境変数 | 🔜 B: `KENNEL_CONFIG`, `KENNEL_NO_COLOR`, `KENNEL_MOCK_SCRIPT`(✅), `KENNEL_APPLE_TESTS`(✅)を文書化 |

### 1.6 フック(ライフサイクルイベント)

Claude Code の hook は「観測」だけでなく「**ブロック・入力書き換え・コンテキスト注入**」ができる。Kennel の `EventBus` は観測専用。

| Claude Code hook | Kennel 相当 | 判定 |
| --- | --- | --- |
| `PreToolUse`(deny/allow/入力更新) | `tool.requested` イベント(観測のみ) | ✅ A: 戻り値で allow/deny/args 更新できる `before_tool` フック(SDK callback) |
| `PermissionRequest` | `prompter` callback | ✅(Approval の戻り値)。✅ A: deny 理由メッセージと args 更新を返せるように |
| `PostToolUse` / `PostToolUseFailure` | `tool.completed` / `tool.failed` | ✅ 観測。🔜 B: 結果を書き換えられる `after_tool` |
| `UserPromptSubmit`(コンテキスト注入/ブロック) | なし | 🔜 B: `before_prompt` |
| `Stop` | `session.completed` | ✅ 観測。🔜 C: 「続行させる」戻り値 |
| `PreCompact` | `context.compacted`(事後) | 🔜 C: 事前フック |
| `SessionStart` / `SessionEnd` | `session.started`/なし | 🔜 C: `session.closed` |
| `Notification`, `SubagentStart/Stop`, `Setup`, `ConfigChange`, `FileChanged`, `CwdChanged`, `TaskCreated/Completed`, `MessageDisplay`, `UserPromptExpansion`, `PostToolBatch`, `PermissionDenied` | — | ❌(`permission.denied` イベントは ✅) |
| 設定ファイルでのシェルコマンド hook(JSON stdin/stdout, exit 2 = block) | なし | 🔜 C: SDK callback を先に安定させ、その後 `"hooks"` 設定でコマンド hook を同じ契約に載せる |

### 1.7 print モードの出力形式

| Claude Code | Kennel |
| --- | --- |
| `text` | ✅ |
| `json`: `result` オブジェクト(`session_id, num_turns, duration_ms, is_error, stop_reason, total_cost_usd, usage, result, structured_output, permission_denials, errors`) | ✅ A: `{"text", "stop_reason", "session_id", "duration_ms", "is_error", "tool_calls":[ToolCallRecord...], "usage", "structured_output", "compactions"}`。費用項目は持たない |
| `stream-json`: `system(init)`, `assistant`, `user`, `result`, `stream_event`, `rate_limit_event`, hook events | ✅ A: `Event` の JSONL(`session.started` を init 相当、`tool.*`, `model.delta`, 最後に `result`)。既存 `--trace` をそのまま stdout へ、スキーマを固定して文書化 |
| exit code | ✅ 0/1/2(設定)/3(モデル不可)/130 |

### 1.8 権限モード・ルール構文

| Claude Code | Kennel |
| --- | --- |
| `default`(都度確認) | ✅ ツール種別ごとの `ask` |
| `acceptEdits` | ✅ `--allow-write` |
| `plan`(読み取り専用) | ✅ `--read-only` |
| `dontAsk`(確認を全部 deny) | ✅ `--non-interactive` |
| `bypassPermissions` | ✅ A: `--permission-mode bypass`(現状は `--allow-write --allow-shell --allow-web` の組合せ) |
| `auto`(分類器) | ❌ |
| ルール `Bash(git *)`, `Read(./src/**)`, 優先順位 deny > ask > allow | ✅ A: `shell(git *)`, `write(docs/**)`, `read(**/*.env)`=deny。優先順位も同じ |
| 「session 中だけ許可」(`a`) | ✅ `Approval.SESSION` |
| 承認したルールの設定ファイルへの書き戻し | 🔜 B(README で「planned」と明記済み。`kennel.local.json` へ) |

### 1.9 セッション

| Claude Code | Kennel |
| --- | --- |
| `~/.claude/projects/<proj>/<id>.jsonl` に永続化 | 🔜 B: `~/.local/state/kennel/sessions/<workspace-hash>/<id>.jsonl`。Apple の `LanguageModelSession` は直列化できないので、履歴(prompt/response/tool_calls)を保存し、resume 時は既存のコンパクション機構で要約を seed する |
| `--continue`, `--resume [id]`, `--session-id`, `--fork-session`, `-n` | 🔜 B |
| resume で復元されるもの(履歴, model, permission mode, goal, tasks) | 🔁: 履歴・permission policy・session grants のみ |
| `/export` | 🔜 B |

### 1.10 拡張点

| Claude Code | Kennel |
| --- | --- |
| カスタムツール = MCP サーバー | ✅ in-process `Tool` サブクラス + `ToolRegistry`(Kennel の方が軽い) |
| `.claude/commands/*.md` | 🔜 C |
| `.claude/agents/*.md`(subagent) | ❌ |
| `.claude/skills/` | ❌ |
| plugins / marketplace | 🔁 C: `kennel.tools` entry point によるツールパッケージ発見のみ |
| MCP サーバー(stdio/http/sse) | 🔜 C: stdio クライアントのみ(local-first の範囲内) |
| output styles, status line | ❌ |

---

## 2. Claude Agent SDK (Python, `claude_agent_sdk` 0.2.154) のインターフェース一覧

### 2.1 トップレベル API

| SDK | 役割 | Kennel |
| --- | --- | --- |
| `query(prompt, options) -> AsyncIterator[Message]` | 一発実行、メッセージを非同期で流す | 🔁 A: `Agent.run()`(✅, 結果を返す)に加えて **`Agent.stream(prompt) -> AsyncIterator[Event]`** を追加。現在 `on_delta` callback と `EventBus`(ワーカースレッドから発火)しか無く、呼び出し側のイベントループで `async for` できない |
| `ClaudeSDKClient` | 双方向・多ターンクライアント | ✅ `Session` が相当 |
| `.connect()/.disconnect()`, async context manager | 接続管理 | ✅ A: `async with agent.new_session() as s:`(`close()` は ✅) |
| `.query()` | 送信 | ✅ `Session.run()` |
| `.receive_messages()/.receive_response()` | 受信 | ✅ A: `Session.stream(prompt)` |
| `.interrupt()` | 実行中ターンの中断 | ✅ A: `Session.interrupt()`(CLI は task cancel で実装済み。SDK 公開 API にする) |
| `.set_permission_mode()` | 実行中に権限変更 | ✅ `agent.permissions.set_decision()`。🔜 B: `Session.set_permission_mode()` 糖衣 |
| `.set_model()` | モデル切替 | ❌(provider 固定。複数 provider 後に検討) |
| `.get_server_info()` | 利用可能ツール・モード等 | ✅ `Session.status()` + `agent.provider.info` |
| `.get_context_usage()` | コンテキスト使用量 | ✅ A: `Session.context_usage()`。4k 窓のオンデバイスでは最重要の可観測性 |
| `.rewind_files()` + `enable_file_checkpointing` | 変更ファイルを巻き戻す | 🔜 C: write/edit 前のスナップショット(`.kennel/checkpoints/`)と `Session.rewind()` |
| `.get_mcp_status()/.reconnect_mcp_server()/.toggle_mcp_server()`, `.stop_task()` | MCP/タスク | ❌ |
| セッションストア関数 `list_sessions`, `get_session_info`, `get_session_messages`, `delete_session`, `rename_session`, `tag_session`, `fork_session`, `SessionStore`/`InMemorySessionStore` | 永続セッション操作 | 🔜 B: `SessionStore` 抽象 + `FileSessionStore`/`InMemorySessionStore`、`kennel.sessions.list/get/delete/fork` |
| `tool()` デコレータ, `create_sdk_mcp_server()`, `SdkMcpTool` | カスタムツール | 🔁 B: `@kennel.tool(name, description)` で関数の型ヒントから `ToolParameter` を生成(spec §6「decorator sugar は v0.2」)。MCP サーバー化は不要 |

### 2.2 `ClaudeAgentOptions` のフィールドと Kennel の対応

| SDK フィールド | Kennel | 判定 |
| --- | --- | --- |
| `cwd` | `Agent(workspace=)` | ✅ |
| `tools`, `allowed_tools`, `disallowed_tools` | `Agent(tools=[...])` | ✅ / ✅ A: `permissions` にルール構文 |
| `system_prompt`(文字列 or `{"type":"preset","preset":"claude_code","append":...}`) | `Agent(instructions=)` は既定指示への**追記** | ✅ A: `instructions=`(追記)と `system_prompt=`(差し替え)を明確に分ける |
| `permission_mode` | 個別 `permissions={}` | ✅ A: `permission_mode="read-only"|...` 糖衣 |
| `can_use_tool(tool_name, input, context) -> Allow(updated_input)/Deny(message, interrupt)` | `prompter(request) -> Approval` | 🔁 A: `prompter` は `ask` 時のみ。全ツール呼び出しを見られる `before_tool` フックと、`Approval` に `updated_arguments`/`message` を持たせる |
| `hooks: dict[HookEvent, list[HookMatcher]]`(PreToolUse, PostToolUse, PostToolUseFailure, UserPromptSubmit, Stop, SubagentStop, PreCompact, Notification, SubagentStart, PermissionRequest) | `events=EventBus` | ✅ A: `Agent(hooks={"before_tool": [...], "after_tool": [...], "before_prompt": [...]})`、matcher はツール名 |
| `max_turns` | なし | ❌(`max_tool_calls` ✅ が相当) |
| `max_budget_usd` | なし | ❌ 費用なし。`turn_timeout_seconds` ✅ |
| `model`, `fallback_model`, `effort`, `thinking`, `max_thinking_tokens`, `betas` | `provider=` | 🔁: `provider=` のみ。`AppleProvider(deterministic=True)` ✅ が「生成プロファイル」に相当 |
| `output_format={"type":"json_schema","schema":...}` → `ResultMessage.structured_output` | `ProviderSession.respond_structured()` | ✅ A: `Agent.run(prompt, schema=...)` / `AgentResult.structured_output` に昇格 |
| `continue_conversation`, `resume`, `session_id`, `fork_session`, `resume_session_at`, `session_store`, `session_store_flush` | `Session(session_id=)` のみ | 🔜 B |
| `env` | `Agent(environment=)` | ✅ |
| `add_dirs` | なし | ❌ v0.1 |
| `mcp_servers`, `strict_mcp_config` | なし | 🔜 C |
| `agents: dict[str, AgentDefinition]` | なし | ❌ |
| `settings`, `setting_sources` | `config=KennelConfig` + `load_config()` | ✅。🔜 B: `setting_sources=("user","project")` |
| `sandbox: SandboxSettings` | なし | ❌ v0.1(spec 非目標)。将来 `shell` に seatbelt |
| `plugins`, `skills` | なし | ❌ |
| `include_partial_messages` | `on_delta` | ✅(`stream()` で統一) |
| `include_hook_events`, `forward_subagent_text` | — | ❌ |
| `permission_prompt_tool_name` | `prompter` | ✅ |
| `stderr`, `debug_stderr` | `--trace`/logging | ✅ |
| `user`, `cli_path`, `extra_args`, `max_buffer_size`, `load_timeout_ms`, `task_budget`, `enable_file_checkpointing` | — | ❌(checkpointing は 🔜 C) |

### 2.3 メッセージ型

| SDK | Kennel | 判定 |
| --- | --- | --- |
| `UserMessage`, `AssistantMessage`, `SystemMessage(subtype, data)`, `ResultMessage`, `StreamEvent`, `RateLimitEvent`, `ConversationResetMessage` | `Event(type, session_id, data, timestamp)`(dict ベース) + `AgentResult` | 🔁 B: `Event` は残し、`stream()` が返す型として `TextDelta`, `ToolCall`, `PermissionRequest`, `Result` 程度の typed dataclass を追加。JSON スキーマを固定 |
| `ResultMessage`(`subtype, duration_ms, duration_api_ms, is_error, num_turns, session_id, stop_reason, total_cost_usd, usage, result, structured_output, model_usage, permission_denials, errors`) | `AgentResult(text, stop_reason, tool_calls, usage, session_id)` | ✅ A: `duration_ms`, `is_error`, `structured_output`, `permission_denials`(tool_calls の status=denied から導出可)を追加 |
| content blocks `TextBlock`, `ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock` | `ToolCallRecord` | ✅ 十分(Apple FM に thinking は無い) |

### 2.4 権限型

| SDK | Kennel |
| --- | --- |
| `PermissionResultAllow(updated_input, updated_permissions)` / `PermissionResultDeny(message, interrupt)` | ✅ A: `Approval` enum を残しつつ `Allow(updated_arguments=, remember=once/session)` / `Deny(message=, interrupt=)` を返せるように |
| `ToolPermissionContext(suggestions, tool_use_id, blocked_path, decision_reason, ...)` | ✅ `PermissionRequest(kind, summary, details, warnings)`。🔜 B: `suggestions`(承認すると書き戻されるルール候補) |
| `PermissionUpdate(type, rules, behavior, mode, directories, destination)` | ✅ `set_decision`, `grant_session`。🔜 B: ルール + `destination`(session/local/project/user) |
| `PermissionMode` literal | ✅ A |

### 2.5 エラー型

| SDK | Kennel |
| --- | --- |
| `ClaudeSDKError` > `CLINotFoundError`, `CLIConnectionError`, `ProcessError` > `ResultError`, `CLIJSONDecodeError` | ✅ `KennelError` > `ModelUnavailableError`, `ProviderError`, `ContextLimitError`, `SessionError`, `PermissionDeniedError`, `ConfigurationError`, `ToolError`(Argument/Execution/OutputLimit), `WorkspaceError`(Escape)。Kennel の方が細かく、プロセス境界が無いので CLI 系は不要。🔜 B: `TurnCancelledError`(interrupt 用) |

### 2.6 その他 export

| SDK | Kennel |
| --- | --- |
| `AgentDefinition` | ❌ |
| `SettingSource` | 🔜 B |
| `SandboxSettings`, `SandboxNetworkConfig` | ❌ v0.1 |
| `McpServerConfig`(stdio/sse/http/sdk) | 🔜 C(stdio のみ) |
| `HookMatcher(matcher, hooks, timeout)`, `HookContext`, `HookJSONOutput`, `*HookInput` | ✅ A(§2.2 hooks) |
| `ContextUsageResponse` | ✅ A |
| `Transport`(カスタムトランスポート) | ✅ `ModelProvider` が相当 |
| `__version__` | ✅ |

---

## 3. Kennel / KennelSDK が備えるべきインターフェース(まとめ)

### 3.1 既に備えているもの(v0.0.3)

- CLI: `kennel [workspace]`, `-p`, `--read-only/--allow-write/--allow-shell/--allow-web`, `--non-interactive`, `--max-tool-calls`, `--verbose`, `--trace`, `--version`; `/help /status /clear /exit`; exit code 0/1/2/3/130。
- SDK: `Agent(workspace, tools, permissions, provider, instructions, config, prompter, events, environment, registry)`, `Session.run/clear/close/status`, `AgentResult`, `ToolCallRecord`, `Tool`/`ToolRegistry`, `PermissionManager`/`Approval`, `ModelProvider`/`ProviderSession`(`respond_structured` 含む), `EventBus`(14 イベント), `KennelConfig`/`load_config`, エラー階層, `context.chunk_text/map_reduce`。
- 設定: `./kennel.json`, `~/.config/kennel/settings.json`。

### 3.2 優先度 A(着地済み: PR #11〜#20 → main `329ef46`)

| # | インターフェース | 根拠 |
| --- | --- | --- |
| A1(PR #11) | `Agent.stream()` / `Session.stream()`(`AsyncIterator`)と `async with` 対応 | SDK が本体。`query()`/`receive_messages()` 相当が無いと app 埋め込みで `on_delta` + スレッド越し EventBus を扱わされる |
| A2(PR #15) | `Session.interrupt()`、`TurnCancelledError` | CLI の Ctrl-C を SDK に昇格 |
| A3(PR #19) | `Session.context_usage()` と `/usage` | 4k 窓が最大の制約。SDK の `get_context_usage` 相当 |
| A4(PR #13) | `--output-format json|stream-json`、`AgentResult` に `duration_ms/is_error/structured_output` | 既存 `--trace` を stdout に出してスキーマ固定するだけで stream-json になる |
| A5(PR #18) | `--json-schema` / `Agent.run(prompt, schema=)` | `respond_structured` を SDK の一級 API に |
| A6(PR #17) | `permission_mode`(`read-only/default/accept-edits/dont-ask/bypass`)と ルール構文 `tool(specifier)` | Claude Code の権限モデルで最も使われる二つ。既存フラグは糖衣化 |
| A7(PR #20) | `before_tool`/`after_tool`/`before_prompt` フック(`Agent(hooks=)`)、`Approval` に `updated_arguments`/`message` | PreToolUse/PostToolUse/UserPromptSubmit/can_use_tool 相当。DOGGIES 等の埋め込み側がポリシーを持てる |
| A8(PR #12) | `--instructions`/`--system-prompt`(CLI)と `system_prompt=`(SDK)の追記・差し替え分離 | 設定にはあるが CLI に無い。追記か差し替えかが今は暗黙 |
| A9(PR #14) | `/compact`, `/permissions` | 内部に実装済み(`_compact`, `set_decision`) |
| A10(PR #16) | `kennel doctor` | 要件が厳しい(Xcode, Apple Intelligence, macOS 26+)ので診断コマンドの価値が高い |

### 3.3 優先度 B(永続化と設定の成熟)

| # | インターフェース |
| --- | --- |
| B1 | 永続セッション: `SessionStore`(File/InMemory)、`--continue/--resume/--session-id/--fork-session/--name/--no-session-persistence`、`/resume /fork /export`、`kennel sessions list/rm`、`kennel.sessions.*` 関数 |
| B2 | 承認ルールの書き戻し(`kennel.local.json`)、`PermissionRequest.suggestions`、`PermissionUpdate.destination` |
| B3 | `KENNEL.md`(project/user)、`/init`、`/memory` |
| B4 | `--config FILE|JSON`, `--no-user-config`/`setting_sources`, `--turn-timeout`, `--no-compact`, `--provider` 公開 |
| B5 | `ask_user` ツール(AskUserQuestion 相当。SDK は prompter 同型の callback) |
| B6 | `@kennel.tool` デコレータ |
| B7 | typed イベント型(`TextDelta`, `ToolCall`, `Result` ...)と stream-json スキーマ文書 |
| B8 | 環境変数の文書化(`KENNEL_CONFIG`, `KENNEL_NO_COLOR`, `KENNEL_MOCK_SCRIPT`, `KENNEL_APPLE_TESTS`) |

### 3.4 優先度 C(将来拡張。architecture を壊さない範囲で)

MCP stdio クライアント bridge(`--mcp-config`, `/mcp`)、設定ファイルのコマンド hook、`web_fetch` 分離、ファイルチェックポイントと `Session.rewind()`、`.kennel/commands/*.md`、entry-point ツールプラグイン、`--trace-file`、`--input-format stream-json`、`/config`、`/model`(複数 provider 後)。

### 3.5 意図的に持たないもの

バックグラウンド/クラウド/リモート系(`--bg`, `agents`, `--cloud`, `--remote-control`, `--teleport`, `--worktree`, `--tmux`)、認証系、subagent(`Agent` ツール、`--agents`、`AgentDefinition`、SubagentStart/Stop hook)、タスク/cron ツール、IDE・Chrome 連携、plugin marketplace、skills、output style/status line、`auto` 権限モード、コスト系(`--max-budget-usd`, `/cost`, `total_cost_usd`)、モデルの effort/thinking/fallback、`--add-dir`(v0.1)、OS サンドボックス(v0.1)。理由はすべて spec §1.1 の非目標か、オンデバイス単一モデル・単一 workspace の前提に反するため。
