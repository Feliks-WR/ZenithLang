#!/usr/bin/env bash
set -euo pipefail

# Regenerate ANTLR4 C++ parser sources for grammar/Zenith.g4
# Usage: scripts/regen_parser.sh [OUT_DIR]

OUT_DIR=${1:-"${PWD}/build/generated/antlr"}
ANTLR_VERSION=4.11.1
ANTLR_JAR_NAME=antlr-${ANTLR_VERSION}-complete.jar
ANTLR_JAR_URL="https://www.antlr.org/download/${ANTLR_JAR_NAME}"
TOOLS_DIR="${PWD}/tools"
JAR_PATH="${TOOLS_DIR}/${ANTLR_JAR_NAME}"

mkdir -p "${TOOLS_DIR}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${JAR_PATH}" ]]; then
  echo "Downloading ANTLR ${ANTLR_VERSION} to ${JAR_PATH}..."
  curl -L -o "${JAR_PATH}" "${ANTLR_JAR_URL}"
fi

# Optional: verify SHA256 if you keep a .sha256 file
# If you want to pin a checksum, create tools/${ANTLR_JAR_NAME}.sha256
if [[ -f "${TOOLS_DIR}/${ANTLR_JAR_NAME}.sha256" ]]; then
  echo "Verifying checksum..."
  sha256sum -c "${TOOLS_DIR}/${ANTLR_JAR_NAME}.sha256"
fi

echo "Generating C++ parser into ${OUT_DIR}..."
java -jar "${JAR_PATH}" -Dlanguage=Cpp -visitor -o "${OUT_DIR}" grammar/Zenith.g4

echo "Done. Generated files are in ${OUT_DIR}."

