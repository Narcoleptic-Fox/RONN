#![allow(missing_docs)]

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use protox (pure Rust protobuf compiler) instead of protoc
    let file_descriptor_set = protox::compile(["proto/onnx.proto"], ["proto/"])?;

    // Compile ONNX protobuf definitions
    prost_build::Config::new()
        .out_dir("src/generated")
        .compile_fds(file_descriptor_set)?;

    // Tell Cargo to rerun build script if proto files change
    println!("cargo:rerun-if-changed=proto/onnx.proto");

    Ok(())
}
