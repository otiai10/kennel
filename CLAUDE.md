@docs/constitution.md

# Kennel — 作業規約

Kennel は Apple Foundation Models 向けのローカルエージェント runtime / Python SDK / CLI。設計判断は上で読み込んだ憲章の 5 原則から導く。文書の置き場は `docs/README.md`。

## 作業原則

- 文書とコードが食い違ったらコードが正しい。文書側を直す
- 機能は SDK(`src/kennel/`)に置き、CLI(`src/kennel/cli/`)はそれを呼ぶだけにする
- 外部消費者向けの契約(stdout の JSON、公開 API の `__all__`)を変えるときは、文書(`docs/output-format.md`)とテストを同じ変更で更新する
- 勝手に実装しない。Issue の受け入れ条件に無いことは越境・残置判断として PR 本文に書く

## 検証

```bash
uv sync --extra dev
uv run ruff check src tests examples
uv run pytest                              # MockProvider ベース。モデル不要
KENNEL_APPLE_TESTS=1 uv run pytest -m apple  # Apple 実機(macOS 26+, Apple Intelligence 有効)
uv run kennel doctor                       # 環境診断
```

apple_fm_sdk の既知の癖(ツール callback は別スレッド、`fm.Tool` はセッション中保持が必要、`respond()` の戻り値型は schema の有無で変わる)は `docs/architecture.md` の Threading 節と `src/kennel/providers/apple.py` の docstring に書いてある。
