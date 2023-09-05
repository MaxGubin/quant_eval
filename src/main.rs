#[allow(unused_imports, unused_variables)]
use clap::Parser;
//use candle_core::quantized::{ggml_file, gguf_file};
//use candle_core::{Device, Tensor};
//use candle_transformers::generation::LogitsProcessor;

mod config;
use config::{Config, Prompts};

mod model;

// =================================================================================================
// Command line arguments.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The configuration file with JSON of the models to load.
    #[arg(short='c', long)]
    config_json: Option<String>,

    /// The configuration file with JSON of the prompts.
    #[arg(short='p', long)]
    prompt_json: Option<String>,
}


fn option_string_to_path(s: &Option<String>) -> anyhow::Result<std::path::PathBuf> {
    match s {
        Some(s) => Ok(std::path::PathBuf::from(s)),
        None => Err(anyhow::anyhow!("No path provided")),
    }
}

impl Args {
    fn load_config(&self) -> anyhow::Result<Config> {
        let config_path = option_string_to_path(&self.config_json)?;
        let config_json = std::fs::read_to_string(config_path)?;
        let config: Config = serde_json::from_str(&config_json)?;
        Ok(config)
    }

    fn load_prompts(&self) -> anyhow::Result<Prompts> {
        let prompt_path = option_string_to_path(&self.prompt_json)?;
        let prompt_json = std::fs::read_to_string(prompt_path)?;
        let prompts: Prompts = serde_json::from_str(&prompt_json)?;
        Ok(prompts)
    }
}
// =================================================================================================

fn process_models(args: &Args) -> anyhow::Result<()> {
    let config = args.load_config()?;
    let prompts = args.load_prompts()?;

    let _model = model::load_model(config)?;
    let _tokenizer = model::load_tokenizer(config)?;

    Ok(())
}


fn main() {
    println!("Processing Data...");
    let args = Args::parse();
    println!("Parsed args: {:?}", args);
    process_models(&args).unwrap();
}
