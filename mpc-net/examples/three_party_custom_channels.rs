use std::path::PathBuf;

use bytes::{Buf, BufMut};
use clap::Parser;
use color_eyre::{eyre::Context, Result};
use futures::{SinkExt, StreamExt};
use mpc_net::{config::NetworkConfig, MpcNetworkHandler};
use tokio_util::codec::{Decoder, Encoder};

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

    let codec = MessageCodec;
    let mut channels = network.get_custom_channels(codec).await?;

    // send to all channels
    for (&i, channel) in channels.iter_mut() {
        let mes = Message::Ping([i as u8; 1024]);
        channel.send(mes).await?;
    }
    // recv from all channels
    for (&_, channel) in channels.iter_mut() {
        let buf = channel.next().await;
        if let Some(Ok(Message::Ping(b))) = buf {
            assert!(b.iter().all(|&x| x == config.my_id as u8))
        } else {
            panic!("could not receive message");
        }
    }
    // send to all channels
    for (&i, channel) in channels.iter_mut() {
        let mes = Message::Pong([i as u8; 512]);
        channel.send(mes).await?;
    }
    // recv from all channels
    for (&_, channel) in channels.iter_mut() {
        let buf = channel.next().await;
        if let Some(Ok(Message::Pong(b))) = buf {
            assert!(b.iter().all(|&x| x == config.my_id as u8))
        } else {
            panic!("could not receive message");
        }
    }
    network.print_connection_stats(&mut std::io::stdout())?;

    Ok(())
}

// A message type that can be sent over the network

#[allow(clippy::large_enum_variant)]
enum Message {
    Ping([u8; 1024]),
    Pong([u8; 512]),
}

impl Message {
    fn id(&self) -> u8 {
        match self {
            Message::Ping(_) => 0,
            Message::Pong(_) => 1,
        }
    }
}

// A codec for the message type
// This is a very inefficient implementation of a codec, but it is just for demonstration purposes

#[derive(Clone, Copy)]
struct MessageCodec;

impl Encoder<Message> for MessageCodec {
    type Error = std::io::Error;

    fn encode(&mut self, item: Message, dst: &mut bytes::BytesMut) -> Result<(), Self::Error> {
        dst.put_u8(item.id());
        match item {
            Message::Ping(buf) => {
                dst.extend_from_slice(&buf);
            }
            Message::Pong(buf) => {
                dst.extend_from_slice(&buf);
            }
        }
        Ok(())
    }
}

impl Decoder for MessageCodec {
    type Item = Message;

    type Error = std::io::Error;

    fn decode(&mut self, src: &mut bytes::BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.is_empty() {
            return Ok(None);
        }
        let id = src[0];
        match id {
            // ping
            0 => {
                if src.len() < 1025 {
                    return Ok(None);
                }
                let mut buf = [0u8; 1024];
                src.advance(1);
                src.copy_to_slice(&mut buf);
                Ok(Some(Message::Ping(buf)))
            }
            // pong
            1 => {
                if src.len() < 512 + 1 {
                    return Ok(None);
                }
                let mut buf = [0u8; 512];
                src.advance(1);
                src.copy_to_slice(&mut buf);
                Ok(Some(Message::Pong(buf)))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid message id",
            )),
        }
    }
}
