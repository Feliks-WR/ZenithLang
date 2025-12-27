#ifndef ZENITH_ZENITHRUNTIME_H
#define ZENITH_ZENITHRUNTIME_H

#include <cstdint>

extern "C" {
    // Standard library functions for Zenith
    void zenith_print_i64(int64_t value);
    void zenith_print_str(const char* value);
    void zenith_print_array(const int64_t* data, int64_t size);
}

#endif // ZENITH_ZENITHRUNTIME_H
