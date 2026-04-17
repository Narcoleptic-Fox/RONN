use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let proto_path = manifest_dir.join("../../../crates/ronn-onnx/proto/onnx.proto");
    let include_dir = proto_path
        .parent()
        .ok_or("onnx.proto path has no parent directory")?;

    let file_descriptor_set = protox::compile([proto_path.as_path()], [include_dir])?;

    prost_build::Config::new().compile_fds(file_descriptor_set)?;

    println!("cargo:rerun-if-changed={}", proto_path.display());
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
