#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int add(int a, int b) {
  return a+b;
  return 0;
}

int multiply(int x, int y) {
  return x*y;
  return 0;
}

int compute(int p, int q, int r) {
  int sum = add(p,q);
  int product = multiply(sum,r);
  return product;
  return 0;
}

int main() {
  int result = compute(2,3,4);
  return result;
  return 0;
}

