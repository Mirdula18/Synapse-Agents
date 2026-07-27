# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `py.typed` marker for PEP 561 type-checking support.
- `.env.example` documenting all 14 environment variables.
- `CHANGELOG.md` (this file).
- `SYNAPSE_ENABLE_EXEC` environment variable (default `false`) — `safe_shell` is
  a no-op unless explicitly enabled. Interim guardrail until full sandbox (F6).

### Changed
- README test-count corrected from 64 to 76.

## [1.0.0] - 2025-07-27

### Added
- Initial release: Planner, Researcher, Executor, Reflector agents.
- FastAPI REST API with synchronous and async task execution.
- SQLite persistent memory with task history and knowledge base.
- Single-file vanilla-JS frontend with async polling.
- CLI mode with interactive plan approval.
- 76 unit tests (all offline, fully mocked).
