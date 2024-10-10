use std::{collections::HashMap, path::PathBuf};

use clap::Parser;
use color_eyre::{eyre::Context, Result};
use mpc_net::{channel::ChannelHandle, config::NetworkConfig, MpcNetworkHandler};

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

    let channels = network.get_byte_channels().await?;
    let mut managed_channels = channels
        .into_iter()
        .map(|(i, c)| (i, ChannelHandle::manage(c)))
        .collect::<HashMap<_, _>>();

    // send to all channels
    for (&i, channel) in managed_channels.iter_mut() {
        let buf = vec![i as u8; 1024];
        let _ = channel.send(buf.into()).await.await?;
    }
    // recv from all channels
    for (&_, channel) in managed_channels.iter_mut() {
        let buf = channel.recv().await.await;
        if let Ok(Ok(b)) = buf {
            println!("received {}, should be {}", b[0], config.my_id as u8);
            assert!(b.iter().all(|&x| x == config.my_id as u8))
        }
    }
    network.print_connection_stats(&mut std::io::stdout())?;

    Ok(())
}
