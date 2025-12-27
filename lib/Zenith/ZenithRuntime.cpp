#include "Zenith/ZenithRuntime.h"
#include <cstdio>
#include <iostream>

extern "C" {

void zenith_write_char(const char value) {
    putchar(value);
}

void zenith_write_str(const char* value) {
    printf("%s", value);
}

void zenith_write_i64(const int64_t value) {
    printf("%ld", value);
}

void zenith_write_array(const int64_t* data, const size_t size) {
    printf("[");
    for (size_t i = 0; i < size; ++i) {
        printf("%ld", data[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]");
}

} // extern "C"
