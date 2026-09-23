# 憲章 — 5 原則

Kennel の設計判断はこの 5 つから導く。文書とコードが食い違ったらコードが正しく、コードと原則が食い違ったらそれは直すべき不整合である。

## 1. ローカル完結 — 既定では推論もデータもデバイスを出ない

Kennel は会議記録のような私的な文書を扱うために存在し、利用者は「Mac の外に何も出ない」ことを前提に使う。オンデバイスモデルを選んだ理由そのものであり、他のすべてに優先する。

- ネットワークに出る機能(`web`)は既定で無効で、明示的に有効化し検索プロバイダを与えたときだけ動く
- Kennel 本体はテレメトリを送らず、バックエンドを持たない
- デバイスの外に出るプロバイダ(Private Cloud Compute 等)は `ProviderInfo.mode` を `local` 以外で申告し、CLI はそれを表示する

正本: `src/kennel/providers/base.py`(`ProviderInfo.mode`)、`src/kennel/tools/web.py`、README「Local-first contract」

## 2. 単一境界 — 同種の判断は 1 箇所だけが下す

同じ判断を複数箇所で実装すると、片方だけ直して抜け道が生まれる。安全性の根拠になる判断ほど 1 箇所に集める。

- パスの解決と workspace 境界の検査は `Workspace.resolve` だけが行う(`..`、絶対パス、symlink を含む)
- すべてのツール呼び出しは `ToolRunner.invoke` を通り、検証・guardrail・フック・権限・実行・出力上限・イベントはそこにしか無い
- 権限の優先順位(deny > specifier > bare、ask > allow、モードは基底層)は `PermissionManager.decision_for` だけが持つ。glob 照合は `rules.py` を `glob` ツールと権限ルールが共用する
- `shell` は workspace 境界を越えられるので、`write` とは別の権限を持ち、`--allow-write` では有効にならない

正本: `src/kennel/workspace.py`、`src/kennel/runner.py`、`src/kennel/permissions.py`、`src/kennel/rules.py`、`tests/unit/test_workspace.py`、`tests/unit/test_permission_rules.py`

## 3. 観測と介入の分離 — 観測は黙り、介入は壊れたら失敗する

「何が起きたか」を伝える経路と「何を許すか」を決める経路を型で分ける。観測が壊れてもエージェントは止めてはならず、ポリシーが壊れたときに黙って素通しさせてはならない。

- `EventBus` の購読者は観測のみ。例外はログされ握りつぶされる。イベントは要約・サイズ・時間だけを運び、ファイル内容や生成テキストは運ばない
- `Hooks`(`before_tool` / `after_tool` / `before_prompt`)と prompter は介入。`before_tool` / `after_tool` が例外を投げたらそのツール呼び出しは失敗する。壊れたポリシーは「ポリシーが無い」ではなく「エラー」である
- フックは同一プロセスのアプリケーションコードと同じ信頼レベルで動く。隔離された拡張点ではない

正本: `src/kennel/events.py`、`src/kennel/hooks.py`、`src/kennel/runner.py`、`src/kennel/session.py`、`tests/unit/test_hooks.py`、`SECURITY.md`

## 4. 推測を実測と混ぜない — プロバイダが測らない数値を Kennel が埋めない

オンデバイス SDK はトークン数を返さない。推定値を実測の形で返すと利用者はそれを信じ、校正する手段も無い。

- `AgentResult.usage` はプロバイダが `ProviderSession.usage()` で実測を返すときだけ埋まり、Apple では `None` のまま
- コンテキスト窓の残量は `Session.context_usage()` が推定してよいが、`ContextUsage.estimated` で推定であることを常に明示する
- 実測が得られるようになったら推定を差し替えるだけで済む形にしておく

正本: `src/kennel/session.py`(`context_usage`、`_reported_usage`)、`src/kennel/providers/base.py`(`Usage`)、`tests/providers/test_mock_agent.py`

## 5. SDK が本体、CLI は consumer — 機能は SDK に置き、CLI は薄い利用者

Kennel は埋め込まれて使われるランタイムであり、ターミナルはその利用者の一つに過ぎない。CLI だけができることがあると、アプリ開発者はそれを再実装させられる。

- ターンの中断(`Session.interrupt()`)、進捗の購読(`Session.stream()`)、コンパクション(`Session.compact()`)、構造化出力(`run(schema=)`)は SDK の公開 API であり、CLI はそれを呼ぶだけ
- CLI のフラグは `Agent(...)` の引数か設定ファイルのキーに 1 対 1 で対応する。優先順位は CLI > `Agent()` 引数 > `./kennel.json` > `~/.config/kennel/settings.json` > 既定
- 外部プロセス向けの契約(`--output-format` の JSON)は文書とテストで固定する(`docs/output-format.md`、`tests/e2e/test_output_format.py`)

正本: `src/kennel/session.py`、`src/kennel/agent.py`、`src/kennel/cli/main.py`、`src/kennel/config.py`
