use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_portable_simd)");
    println!("cargo:rerun-if-env-changed=RUSTC");

    let Some(rustc) = env::var_os("RUSTC") else {
        return;
    };

    let Ok(output) = Command::new(rustc).arg("-vV").output() else {
        return;
    };

    if !output.status.success() {
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let is_nightly = stdout
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .is_some_and(|release| release.contains("nightly"));

    if is_nightly {
        println!("cargo:rustc-cfg=has_portable_simd");
    }
}
