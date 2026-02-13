// Basic arithmetic example in Zenith

func.func @main() {
    %c1 = zenith.constant 10 : i32
    %c2 = zenith.constant 20 : i32

    %sum = zenith.add %c1, %c2 : i32
    zenith.print %sum : i32

    return
}

