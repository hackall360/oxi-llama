use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

pub use envconfig;

#[derive(Parser)]
#[command(name = "ollama", about = "Rust reimplementation of the Ollama CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Create a model from a Modelfile
    Create {
        /// Name of the model to create
        model: String,
        /// Path to a Modelfile
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Quantization option
        #[arg(short, long)]
        quantize: Option<String>,
    },
    /// Start the server
    Serve {},
    /// Show model information
    Show {},
    /// Run a model
    Run {},
    /// Stop a running model
    Stop {},
    /// Pull a model
    Pull {},
    /// Push a model
    Push {},
    /// List models
    List {},
    /// Process status
    Ps {},
    /// Copy a model
    Copy {},
    /// Delete a model
    Delete {},
    /// Runner helpers
    Runner {},
}

/// Execute the CLI based on command line arguments.
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Create { model, file, .. } => create_handler(model, file),
        _ => {
            println!("subcommand not yet implemented");
            Ok(())
        }
    }
}

fn create_handler(model: String, file: Option<PathBuf>) -> Result<()> {
    let filename = file.unwrap_or_else(|| PathBuf::from("Modelfile"));
    if !filename.exists() {
        bail!("specified Modelfile wasn't found");
    }
    // Placeholder for actual model creation logic
    println!("creating model {model} from {:?}", filename);
    Ok(())
}
