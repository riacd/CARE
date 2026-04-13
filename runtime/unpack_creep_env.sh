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

cat > "${TARGET_DIR}/activate_runtime.sh" <<'EOF'
#!/usr/bin/env bash
CARE_CREEP_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARE_CREEP_ENV_DIR
export PYTHONHOME="${CARE_CREEP_ENV_DIR}"
export PATH="${CARE_CREEP_ENV_DIR}/bin:${PATH}"
EOF

chmod +x "${TARGET_DIR}/activate_runtime.sh"

cat <<EOF
environment unpacked to: ${TARGET_DIR}
activate with:
  source ${TARGET_DIR}/activate_runtime.sh
EOF
