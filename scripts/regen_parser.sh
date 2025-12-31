#!/usr/bin/env bash
set -euo pipefail

# Regenerate ANTLR4 C++ parser sources for grammar/ZenithParser.g4 + ZenithLexer.g4
# Usage: scripts/regen_parser.sh [OUT_DIR]

OUT_DIR=${1:-"${PWD}/build/generated/antlr"}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ANTLR_VERSION=4.13.2
ANTLR_JAR_NAME=antlr-${ANTLR_VERSION}-complete.jar
ANTLR_JAR_URL="https://www.antlr.org/download/${ANTLR_JAR_NAME}"
TOOLS_DIR="${REPO_ROOT}/tools"
JAR_PATH="${TOOLS_DIR}/${ANTLR_JAR_NAME}"
PARSER_PATH="${REPO_ROOT}/grammar/ZenithParser.g4"
LEXER_PATH="${REPO_ROOT}/grammar/ZenithLexer.g4"

mkdir -p "${TOOLS_DIR}"
mkdir -p "${OUT_DIR}"

if [[ ! -f "${JAR_PATH}" ]]; then
  echo "Downloading ANTLR ${ANTLR_VERSION} to ${JAR_PATH}..."
  curl -L -o "${JAR_PATH}" "${ANTLR_JAR_URL}"
fi

# Optional: verify SHA256 if you keep a .sha256 file
if [[ -f "${TOOLS_DIR}/${ANTLR_JAR_NAME}.sha256" ]]; then
  echo "Verifying checksum..."
  sha256sum -c "${TOOLS_DIR}/${ANTLR_JAR_NAME}.sha256"
fi

echo "Generating C++ parser into ${OUT_DIR}..."
java -jar "${JAR_PATH}" -Dlanguage=Cpp -visitor -o "${OUT_DIR}" "${LEXER_PATH}" "${PARSER_PATH}"

echo "Done. Generated files are in ${OUT_DIR}."
