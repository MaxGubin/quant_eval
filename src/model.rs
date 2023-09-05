use std::collections::HashMap;

use candle_core::quantized::QTensor;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};

use crate::config::ModelConfig;
use tokenizers::Tokenizer;

struct RmsNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}


struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}


impl ModelWeights {
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        Ok(Self{})
    }

    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
    ) -> Result<Self> {
        Ok(Self{})
    }
}

// =================================================================================================

/// Loads the tokenizer.
pub fn load_tokenizer(config: &ModelConfig) -> anyhow::Result<tokenizers::Tokenizer> {
    if config.tokenizer.is_empty() {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
        api.get("tokenizer.json")?
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&config.tokenizer)?;
    Ok(tokenizer)
}

/// Loads the model.
pub fn load_model(config: &ModelConfig) -> anyhow::Result<ModelWeights> {
    let path = std::path::PathBuf::from(&config.model_path);
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) if ext == "ggml" => load_model_ggml(config),
        Some(ext) if ext == "gguf" => load_model_gguf(config),
        _ => Err(anyhow::anyhow!("Unknown model format")),
    }
    let mut reader = std::io::BufReader::new(std::fs::File::open(&config.model_path)?);
    let ct = gguf_file::Content::read(&mut reader)?;
    ModelWeights::from_gguf(ct, &mut reader)
}