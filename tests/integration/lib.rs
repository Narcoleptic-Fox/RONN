#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(clippy::useless_vec)]
#![allow(clippy::identity_op)]
/// Integration tests library for real ONNX models.
///
/// This library contains end-to-end integration tests using production-quality
/// models from computer vision, NLP, and text generation domains.
///
/// Tests are organized by model type:
/// - test_resnet: ResNet-18 image classification
/// - test_bert: DistilBERT NLP embeddings
/// - test_gpt: GPT-2 Small text generation
pub mod common;

// Re-export test modules
#[cfg(test)]
mod test_resnet;

#[cfg(test)]
mod test_bert;

#[cfg(test)]
mod test_gpt;
