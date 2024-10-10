use std::path::PathBuf;

use clap::Parser;
use color_eyre::{eyre::Context, Result};
use futures::{SinkExt, StreamExt};
use mpc_net::{config::NetworkConfig, MpcNetworkHandler};

#[derive(Parser)]
struct Args {
    /// The config file path
    #[clap(short, long, value_name = "FILE")]
    config_file: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let config: NetworkConfig =
        toml::from_str(&std::fs::read_to_string(args.config_file).context("opening config file")?)
            .context("parsing config file")?;

    let network = MpcNetworkHandler::establish(config.clone()).await?;

    let mut channels = network.get_byte_channels().await?;

    // send to all channels
    for (&i, channel) in channels.iter_mut() {
        let buf = vec![i as u8; 1024];
        channel.send(buf.into()).await?;
    }
    // recv from all channels
    for (&_, channel) in channels.iter_mut() {
        let buf = channel.next().await;
        if let Some(Ok(b)) = buf {
            println!("received {}, should be {}", b[0], config.my_id as u8);
            assert!(b.iter().all(|&x| x == config.my_id as u8))
        }
    }
    network.print_connection_stats(&mut std::io::stdout())?;

    Ok(())
}
