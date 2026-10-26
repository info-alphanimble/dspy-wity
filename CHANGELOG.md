# Changelog

All notable changes to `dspy-wity`. Versions follow [semver](https://semver.org). Before 1.0, a minor version can add features.

## 0.2.0 (2026-10-26)

### Added

- **`wity_tools()`** gives agents like `dspy.ReAct` two tools. The agent runs on any text LLM; Wity does the reading.
  - `wity_split(text, split)` cuts a document into whole items, like emails in a thread. Each item comes back exactly as written.
  - `wity_extract(text, fields)` pulls named fields out of a document. Each field comes back as a list of values, as written, or `[]` when there are none.
  - Both take the same `api_key`, `base_url`, `timeout`, `max_retries` and `http_client` options as `Wity`.
- **Update notice.** At a terminal, `dspy-wity` tells you on stderr when a newer version is on PyPI, at most once a day. The check runs in a background thread, so nothing waits for it. It stays quiet in CI, when output is piped, and on servers. `WITY_NO_UPDATE_CHECK=1` turns it off.

### Changed

- Needs `wity-sdk` 0.2.0 or newer.

## 0.1.0 (2026-10-14)

First release.

- `Wity`: plugs in where DSPy expects an LM, and answers signatures with yes/no, choice or score outputs, with a probability for every option.
- Works with DSPy's own decision types (`Noul`, `Choice`, `Score`) and its `ReAnchor` optimizer.
- `reasoning` modes, `max_latency_ms`, and DSPy's cache, history and usage tracking.
