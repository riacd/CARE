#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-./envs/CREEP}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${SCRIPT_DIR}/CARE_CREEP_env.tar.gz"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "missing archive: ${ARCHIVE_PATH}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${TARGET_DIR}"
"${TARGET_DIR}/bin/conda-unpack"

cat <<EOF
environment unpacked to: ${TARGET_DIR}
activate with:
  source ${TARGET_DIR}/bin/activate
EOF
