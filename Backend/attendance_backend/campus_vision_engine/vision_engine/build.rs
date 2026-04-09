use std::{env, fs, path::{Path, PathBuf}};

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    match target_os.as_str() {
        "windows" => build_windows(),
        "linux"   => build_linux(),
        "macos"   => build_macos(),
        other     => panic!("Unsupported OS: {}", other),
    }
}

//
// ── WINDOWS: Copy DLLs next to executable ─────────────────────────────
//
fn build_windows() {
    // Allow override via env (CI / different machines)
    let opencv_bin = env::var("OPENCV_BIN_DIR")
        .unwrap_or_else(|_| r"C:\tools\opencv\build\x64\vc16\bin".to_string());

    let opencv_bin = Path::new(&opencv_bin);

    let dlls = [
        "opencv_videoio_ffmpeg4110_64.dll",
        "opencv_videoio_msmf4110_64.dll",
        "opencv_world4110.dll",
    ];

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // target/debug or target/release
    let exe_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Failed to determine exe_dir");

    println!("📦 Copying OpenCV DLLs → {:?}", exe_dir);

    let mut copied_any = false;

    for dll in &dlls {
        let src = opencv_bin.join(dll);
        let dest = exe_dir.join(dll);

        if !src.exists() {
            println!("cargo:warning=Missing DLL: {}", src.display());
            continue;
        }

        // Copy only if needed (optimization)
        let should_copy = match (src.metadata(), dest.metadata()) {
            (Ok(s), Ok(d)) => s.modified().ok() > d.modified().ok(),
            _ => true,
        };

        if should_copy {
            fs::copy(&src, &dest)
                .unwrap_or_else(|e| panic!("Failed to copy {}: {}", dll, e));

            println!("cargo:warning=Copied {}", dll);
        }

        copied_any = true;
    }

    if !copied_any {
        println!("cargo:warning=⚠️ No OpenCV DLLs found. Set OPENCV_BIN_DIR.");
    }

    println!("cargo:rerun-if-env-changed=OPENCV_BIN_DIR");
    println!("cargo:rerun-if-changed={}", opencv_bin.display());
}

//
// ── LINUX: Use system OpenCV (pkg-config) ─────────────────────────────
//
fn build_linux() {
    if let Ok(lib_dir) = env::var("OPENCV_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
        println!("cargo:rustc-link-lib=dylib=opencv_core");
        println!("cargo:rustc-link-lib=dylib=opencv_imgproc");
        println!("cargo:rustc-link-lib=dylib=opencv_videoio");
        println!("cargo:rustc-link-lib=dylib=opencv_objdetect");
    }

    println!("cargo:rerun-if-env-changed=OPENCV_LIB_DIR");
}

//
// ── MACOS: Homebrew OpenCV ────────────────────────────────────────────
//
fn build_macos() {
    let brew_prefix = env::var("HOMEBREW_PREFIX")
        .unwrap_or_else(|_| "/opt/homebrew".to_string());

    let lib_dir = format!("{}/lib", brew_prefix);

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");
}