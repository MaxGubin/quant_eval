



// =================================================================================================
// Configuration parameter.

#[derive(Debug, serde::Deserialize)]
pub struct ModelConfig {
    /// The name of the model.
    pub name: String,

    /// The path to the model.
    pub model_path: String,
    
    /// The tokenizer parameters
    pub tokenizer: String,

    /// Model Architecture.
    pub architecture: String,   

}

#[derive(Debug, serde::Deserialize)]
pub struct Config {
    models: Vec<ModelConfig>,
}


#[derive(Debug, serde::Deserialize)]
pub struct OnePrompt {
    /// The text of the prompt.
    prompt: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct Prompts {
    prompts: Vec<OnePrompt>,
}
