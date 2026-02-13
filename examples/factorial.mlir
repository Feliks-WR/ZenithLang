// Example: Recursive factorial computation in Zenith MLIR

func.func @factorial(%n: i32) -> i32 {
    %c0 = zenith.constant 0 : i32
    %c1 = zenith.constant 1 : i32

    // Base case: if n <= 1, return 1
    %cmp = arith.cmpi sle, %n, %c1 : i32
    cf.cond_br %cmp, ^bb_base, ^bb_recurse

^bb_base:
    return %c1 : i32

^bb_recurse:
    // Recursive case: n * factorial(n-1)
    %n_minus_1 = zenith.sub %n, %c1 : i32
    %rec_result = zenith.call @factorial(%n_minus_1) : (i32) -> i32
    %result = zenith.mul %n, %rec_result : i32
    return %result : i32
}

func.func @main() {
    %c5 = zenith.constant 5 : i32
    %result = zenith.call @factorial(%c5) : (i32) -> i32
    zenith.print %result : i32
    return
}

