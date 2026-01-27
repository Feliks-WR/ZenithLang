# CTest Configuration for Zenith Language
# Configures test execution, timeouts, and reporting

# Set test timeout (seconds)
set(CTEST_TEST_TIMEOUT 30)

# Enable verbose output for failed tests
set(CTEST_OUTPUT_ON_FAILURE ON)

# Configure test execution
set(CTEST_PARALLEL_LEVEL 4)

# Test result output format
set(CTEST_USE_LAUNCHERS 1)
