// RUN: zenith-opt %s | zenith-opt | FileCheck %s

// CHECK-LABEL: func @constant_test
func.func @constant_test() {
    // CHECK: %[[C1:.*]] = zenith.constant 42 : i32
    %0 = zenith.constant 42 : i32
    // CHECK: %[[C2:.*]] = zenith.constant 3.14 : f64
    %1 = zenith.constant 3.14 : f64
    return
}

// CHECK-LABEL: func @add_test
func.func @add_test(%arg0: i32, %arg1: i32) -> i32 {
    // CHECK: %[[R:.*]] = zenith.add %arg0, %arg1 : i32
    %0 = zenith.add %arg0, %arg1 : i32
    // CHECK: return %[[R]] : i32
    return %0 : i32
}

// CHECK-LABEL: func @arithmetic_test
func.func @arithmetic_test(%arg0: i32, %arg1: i32) -> i32 {
    %0 = zenith.add %arg0, %arg1 : i32
    %1 = zenith.sub %0, %arg1 : i32
    %2 = zenith.mul %1, %arg0 : i32
    return %2 : i32
}

