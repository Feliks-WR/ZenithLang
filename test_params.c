#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int add(int a, int b) {
  return a+b;
  return 0;
}

int multiply(int x, int y) {
  int result = x*y;
  return result;
  return 0;
}

int main() {
  int sum = add(5,3);
  int product = multiply(4,7);
  int total = add(sum,product);
  return total;
  return 0;
}

