#!/bin/bash
# Moves the venv from local_env/venv/ to .venv/ and fixes paths.
# Run once from the repo root, then activate with: source local_env/activate.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OLD_VENV="${REPO_ROOT}/local_env/venv"
NEW_VENV="${REPO_ROOT}/.venv"

if [[ ! -d "${OLD_VENV}" ]]; then
    echo "ERROR: ${OLD_VENV} does not exist, nothing to migrate."
    exit 1
fi

if [[ -d "${NEW_VENV}" ]]; then
    echo "ERROR: ${NEW_VENV} already exists. Remove it first if you want to re-migrate."
    exit 1
fi

echo "Moving ${OLD_VENV} → ${NEW_VENV} ..."
mv "${OLD_VENV}" "${NEW_VENV}"

# Fix the venv's internal paths (pyvenv.cfg, bin/activate*, bin/pip, etc.)
# The key file is pyvenv.cfg — it just has the home dir for the base python, no change needed.
# But the shebang lines in bin/* and the VIRTUAL_ENV variable in activate scripts need updating.

echo "Patching activate scripts..."
sed -i "s|${OLD_VENV}|${NEW_VENV}|g" "${NEW_VENV}/bin/activate"
sed -i "s|${OLD_VENV}|${NEW_VENV}|g" "${NEW_VENV}/bin/activate.csh" 2>/dev/null || true
sed -i "s|${OLD_VENV}|${NEW_VENV}|g" "${NEW_VENV}/bin/activate.fish" 2>/dev/null || true

echo "Patching shebangs in bin/ scripts..."
find "${NEW_VENV}/bin" -type f -exec grep -l "${OLD_VENV}" {} \; | while read -r f; do
    sed -i "s|${OLD_VENV}|${NEW_VENV}|g" "$f"
done

echo ""
echo "Done! Activate with:"
echo "  source local_env/activate.sh"
echo ""
echo "The activation script already points to .venv/ (updated in local_env_build.sh)."
