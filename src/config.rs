



// =================================================================================================
// Configuration parameter.


#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    /// The name of the model.
    pub name: String,

    /// The path to the model.
    pub model_path: String,
    
    /// The tokenizer parameters
    #[serde(default)]
    pub tokenizer: String,

    /// Model Architecture.
    #[serde(default)]
    pub architecture: String,   

    /// Max sequence length.
    #[serde(default="max_seq_len_defalt")]
    pub max_seq_len: usize,
}

fn max_seq_len_defalt() -> usize {
    4096
}

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    pub models: Vec<ModelConfig>,
}


#[derive(Debug, serde::Deserialize)]
pub struct OnePrompt {
    /// The text of the prompt.
    pub prompt: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct Prompts {
    pub prompts: Vec<OnePrompt>,
}
