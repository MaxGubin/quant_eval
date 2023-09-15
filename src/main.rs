use std::{any, sync::Arc};

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
#[allow(unused_imports, unused_variables)]
use clap::Parser;
use std::io::Write;

mod config;
use config::{Config, Prompts};
use tokenizers::Model;

mod model;

// =================================================================================================
// Command line arguments.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The configuration file with JSON of the models to load.
    #[arg(short = 'c', long, default_value = "config/models.json")]
    config_json: Option<String>,

    /// The configuration file with JSON of the prompts.
    #[arg(short = 'p', long, default_value = "config/prompts.json")]
    prompt_json: Option<String>,

    /// Seed for random number generator.
    #[arg(long, default_value = "42")]
    seed: Option<u64>,

    /// Generation temperature.
    #[arg(long, default_value = "0.0")]
    temperature: Option<f64>,

    /// The length of sampled data in tokens (including prompt).
    #[arg(long, default_value = "512")]
    sample_length: Option<usize>,
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

fn decode_ascii(text: &String) -> Option<char> {
    text.strip_prefix("<0x")
        .and_then(|s| s.strip_suffix(">"))
        .and_then(|s| u32::from_str_radix(s, 16).ok())
        .and_then(|c| std::char::from_u32(c))
}

fn print_token(tokenizer: &tokenizers::Tokenizer, token: u32) {
    match tokenizer.id_to_token(token) {
        Some(text) => {
            let text = text.replace("▁", " ");
            match decode_ascii(&text) {
                Some(c) => print!("{}", c),
                None => print!("{}", text),
            }
        }
        None => print!("<UNK>"),
    }
    std::io::stdout().flush();
}
fn process_models(args: &Args) -> anyhow::Result<()> {
    let config = args.load_config()?;
    let prompts = args.load_prompts()?;

    for mc in config.models {
        println!("Loading model: {}", mc.name);
        let mut model = model::load_model(&mc)?;
        let tokenizer = model::load_tokenizer(&mc)?;
        for p in prompts.prompts.iter() {
            println!("Prompt: {}", p.prompt);
            // TODO(ngubin): figure out how to avoid clone here.
            let tokens = tokenizer
                .encode(p.prompt.clone(), true)
                .map_err(anyhow::Error::msg)?;
            let prompt_tokens = tokens.get_ids().to_vec();
            let mut logits_processor =
                LogitsProcessor::new(args.seed.unwrap_or_else(|| 42), args.temperature);
            let to_sample = args
                .sample_length
                .unwrap_or_else(|| 100)
                .saturating_sub(prompt_tokens.len());

            // The first token is generated  a different way as we want to process all prompt tokens in parallel.
            // This token will be used in the loop.
            let mut next_token = {
                let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
                let logits = model.forward(&input, 0)?;
                let logits = logits.squeeze(0)?;
                logits_processor.sample(&logits)?
            };

            for index in 0..to_sample {
                let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
                let logits = model.forward(&input, prompt_tokens.len() + index)?;
                let logits = logits.squeeze(0)?;
                next_token = logits_processor.sample(&logits)?;
                print_token(&tokenizer, next_token);
            }
        }
    }

    Ok(())
}

fn main() {
    println!("Processing Data...");
    let args = Args::parse();
    println!("Parsed args: {:?}", args);
    process_models(&args).unwrap();
}
