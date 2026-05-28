#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${LINEAR_API_KEY:-}" ]]; then
  printf 'LINEAR_API_KEY is required in the environment.\n' >&2
  exit 1
fi

TRIMMED_LINEAR_API_KEY="$(printf '%s' "${LINEAR_API_KEY}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [[ -z "${TRIMMED_LINEAR_API_KEY}" ]]; then
  printf 'LINEAR_API_KEY is empty after trimming whitespace.\n' >&2
  exit 1
fi
if [[ "${TRIMMED_LINEAR_API_KEY}" != "${LINEAR_API_KEY}" ]]; then
  printf 'LINEAR_API_KEY had surrounding whitespace; using the trimmed value for this process.\n' >&2
  export LINEAR_API_KEY="${TRIMMED_LINEAR_API_KEY}"
fi

if ! command -v codex >/dev/null 2>&1; then
  printf 'codex is required on PATH for codex app-server.\n' >&2
  exit 1
fi

SYMPHONY_BIN="${SYMPHONY_BIN:-symphony}"
if [[ "${SYMPHONY_BIN}" == */* ]]; then
  SYMPHONY_PATH="${SYMPHONY_BIN}"
elif command -v "${SYMPHONY_BIN}" >/dev/null 2>&1; then
  SYMPHONY_PATH="$(command -v "${SYMPHONY_BIN}")"
else
  SYMPHONY_PATH=""
fi

if [[ -z "${SYMPHONY_PATH}" || ! -x "${SYMPHONY_PATH}" ]]; then
  printf 'Symphony executable not found. Set SYMPHONY_BIN=/path/to/symphony or add symphony to PATH.\n' >&2
  exit 1
fi

LOGS_ROOT="${SYMPHONY_LOGS_ROOT:-${REPO_ROOT}/.symphony/logs}"
mkdir -p "${LOGS_ROOT}"

args=(
  "--i-understand-that-this-will-be-running-without-the-usual-guardrails"
  "${REPO_ROOT}/WORKFLOW.md"
  "--logs-root"
  "${LOGS_ROOT}"
)
if [[ -n "${SYMPHONY_PORT:-}" ]]; then
  args+=("--port" "${SYMPHONY_PORT}")
fi

SYMPHONY_DIR="$(cd "$(dirname "${SYMPHONY_PATH}")/.." && pwd)"
if ! command -v escript >/dev/null 2>&1 && [[ -f "${SYMPHONY_DIR}/mise.toml" ]] && command -v mise >/dev/null 2>&1; then
  cd "${SYMPHONY_DIR}"
  exec mise exec -- "${SYMPHONY_PATH}" "${args[@]}"
fi

exec "${SYMPHONY_PATH}" "${args[@]}"
