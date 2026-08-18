//! `skadoosh` binary: parse CLI → init tracing → dispatch.

use std::path::Path;
use std::process::ExitCode;

use skadoosh::audio::input::list_devices;
use skadoosh::{Config, Pipeline, Result};
use tracing_subscriber::EnvFilter;

fn main() -> ExitCode {
    let config = Config::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    match dispatch(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!(%err, "skadoosh exited with an error");
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(config: Config) -> Result<()> {
    config.validate()?;

    if config.list_devices {
        for name in list_devices()? {
            println!("{name}");
        }
        return Ok(());
    }

    let selftest = config.selftest.clone();
    let pipeline = Pipeline::new(config)?;
    match selftest {
        Some(wav) => {
            let report = pipeline.run_selftest(&wav, Path::new("selftest_out.wav"))?;
            println!("{report:#?}");
            Ok(())
        }
        None => pipeline.run(),
    }
}
