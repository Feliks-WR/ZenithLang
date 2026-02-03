#!/bin/bash
# ANTLR4 compatibility patches - run after cmake configure

cd "$(dirname "$0")/build/antlr_gen" || { echo "Error: antlr_gen not found"; exit 1; }

echo "Patching ANTLR4 files..."

# Remove override keywords
sed -i 's/ override//g' *.h *.cpp 2>/dev/null

# Remove getSerializedATN (incompatible return type)
sed -i '/virtual.*getSerializedATN/d' *.h 2>/dev/null
sed -i '/^const std::vector<uint16_t>.*getSerializedATN/,/^}/d' *.cpp 2>/dev/null

# Fix string_view conversions  
sed -i 's/std::string name = _vocabulary\.getLiteralName/std::string name(std::string(_vocabulary.getLiteralName/g' *.cpp 2>/dev/null
sed -i 's/getLiteralName(i);/getLiteralName(i)));/g' *.cpp 2>/dev/null

# Comment out ATN deserialization (needs modern runtime or manual fix)
sed -i 's/^\(\s*\)_atn = deserializer\.deserialize/  \/\/ _atn = deserializer.deserialize/' *.cpp 2>/dev/null

echo "✓ Done - build may still need modern ANTLR4 runtime or grammar regeneration"
