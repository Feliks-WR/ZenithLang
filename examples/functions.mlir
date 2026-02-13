// Function call example in Zenith

func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %result = zenith.add %arg0, %arg1 : i32
    return %result : i32
}

func.func @multiply(%arg0: i32, %arg1: i32) -> i32 {
    %result = zenith.mul %arg0, %arg1 : i32
    return %result : i32
}

func.func @main() {
    %c1 = zenith.constant 5 : i32
    %c2 = zenith.constant 3 : i32

    %sum = zenith.call @add(%c1, %c2) : (i32, i32) -> i32
    %product = zenith.call @multiply(%c1, %c2) : (i32, i32) -> i32

    zenith.print %sum : i32
    zenith.print %product : i32

    return
}

