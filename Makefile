# Makefile for convenient Zenith development tasks

.PHONY: help build test clean install run-tests coverage

help:
	@echo "Zenith Language Compiler - Development Tasks"
	@echo ""
	@echo "Available targets:"
	@echo "  build        - Build the compiler"
	@echo "  test         - Run all tests"
	@echo "  clean        - Clean build artifacts"
	@echo "  run-tests    - Run tests with verbose output"
	@echo "  install-deps - Install dependencies (Ubuntu/Debian)"
	@echo "  format       - Format C++ code"

build:
	cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
	cmake --build build

test:
	cd build && ctest --output-on-failure

run-tests:
	cd build && ctest --output-on-failure --verbose

clean:
	rm -rf build cmake-build-debug

install-deps:
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		cmake \
		ninja-build \
		gcc \
		g++ \
		libgtest-dev \
		libantlr4-runtime-dev \
		antlr4

format:
	find src include tests -name "*.cpp" -o -name "*.h" | xargs clang-format -i

ci-test: build test
	@echo "✓ CI tests passed"
