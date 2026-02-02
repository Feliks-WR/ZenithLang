#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int safeDivide(int numerator, int denominator) {
  return numerator/denominator;
  return 0;
}

int validatePercentage(int value) {
  return value;
  return 0;
}

int square(int x) {
  return x*x;
  return 0;
}

int getRandomInRange() {
  return 5;
  return 0;
}

int buildVector(int x, int y, int z) {
  return x+y+z;
  return 0;
}

int main() {
  int divisor = 0;
  int percentage = 0;
  int positive = 0;
  divisor = 5;
  percentage = 75;
  positive = 10;
  int division_result = safeDivide(20,divisor);
  int validated = validatePercentage(percentage);
  int squared = square(positive);
  int vector_sum = buildVector(1,2,3);
  printf("%d\n", division_result);
  printf("%d\n", validated);
  printf("%d\n", squared);
  printf("%d\n", vector_sum);
  return 0;
  return 0;
}

