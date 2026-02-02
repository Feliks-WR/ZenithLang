#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int test_arithmetic() {
  int a = 10;
  int b = 20;
  int sum = a+b;
  return sum;
  return 0;
}

int test_conditional() {
  int x = 15;
  if (x>10) {
  return 1;
  }
  return 0;
  return 0;
}

int test_loop() {
  int i = 0;
  int total = 0;
  while (i<5) {
  total = total+i;
  i = i+1;
  }
  return total;
  return 0;
}

int main() {
  int result1 = test_arithmetic();
  int result2 = test_conditional();
  int result3 = test_loop();
  return 0;
  return 0;
}

