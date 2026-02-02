#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int divide(int a, int b) {
  return a/b;
  return 0;
}

int getScore(int x) {
  return x;
  return 0;
}

int getPositive() {
  return 42;
  return 0;
}

int main() {
  int x = 0;
  int y = 0;
  x = 10;
  y = 85;
  int result = divide(x,3);
  int positive = getPositive();
  int test_score = getScore(y);
  printf("%d\n", result);
  printf("%d\n", positive);
  printf("%d\n", test_score);
  return 0;
  return 0;
}

