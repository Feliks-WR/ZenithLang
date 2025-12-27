use std::path::PathBuf;
use std::env;

fn main() {
    // Get the project manifest dir to resolve relative paths correctly
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Check multiple possible build directories
    let possible_paths = vec![
        PathBuf::from(&manifest_dir).join("../../build"),
        PathBuf::from(&manifest_dir).join("../../cmake-build-debug"),
        PathBuf::from(&manifest_dir).join("../../cmake-build-release"),
    ];

    let mut found = false;
    for lib_path in possible_paths {
        if lib_path.exists() {
            let lib_path_str = lib_path.to_string_lossy().to_string();
            println!("cargo:rustc-link-search=native={}", lib_path_str);
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path_str);
            found = true;
            eprintln!("Found ZenithDialect library at: {}", lib_path_str);
            break;
        }
    }

    if !found {
        eprintln!("WARNING: Could not find build directory with ZenithDialect library.");
        eprintln!("Please build the C++ library first:");
        eprintln!("  cd /home/abdullah/CLionProjects/Zenith");
        eprintln!("  mkdir -p build && cd build");
        eprintln!("  cmake .. && make ZenithDialect");
    }

    // Link against the ZenithDialect library (which contains ZenithBridge.cpp)
    println!("cargo:rustc-link-lib=dylib=ZenithDialect");

    // Rerun if the library changes
    println!("cargo:rerun-if-changed=../../lib/Zenith/ZenithBridge.cpp");
    println!("cargo:rerun-if-changed=../../lib/Zenith/ZenithDialect.cpp");
}
