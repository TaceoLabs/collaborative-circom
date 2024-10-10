//! A simple networking layer for MPC protocols.
#![warn(missing_docs)]
use std::{
    collections::{BTreeMap, HashMap},
    io,
    net::{SocketAddr, ToSocketAddrs},
    sync::Arc,
    time::Duration,
};

use channel::{BytesChannel, Channel};
use codecs::BincodeCodec;
use color_eyre::eyre::{self, Context, Report};
use config::NetworkConfig;
use quinn::{
    crypto::rustls::QuicClientConfig,
    rustls::{
        pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer},
        RootCertStore,
    },
};
use quinn::{
    ClientConfig, Connection, Endpoint, IdleTimeout, RecvStream, SendStream, TransportConfig,
    VarInt,
};
use serde::{de::DeserializeOwned, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_util::codec::{Decoder, Encoder, LengthDelimitedCodec};

pub mod channel;
pub mod codecs;
pub mod config;

/// A network handler for MPC protocols.
#[derive(Debug)]
pub struct MpcNetworkHandler {
    // this is a btreemap because we rely on iteration order
    connections: BTreeMap<usize, Connection>,
    endpoints: Vec<Endpoint>,
    my_id: usize,
}

impl MpcNetworkHandler {
    /// Tries to establish a connection to other parties in the network based on the provided [NetworkConfig].
    pub async fn establish(config: NetworkConfig) -> Result<Self, Report> {
        config.check_config()?;
        let certs: HashMap<usize, CertificateDer> = config
            .parties
            .iter()
            .map(|p| {
                let cert = std::fs::read(&p.cert_path)
                    .with_context(|| format!("reading certificate of party {}", p.id))?;
                Ok((p.id, CertificateDer::from(cert)))
            })
            .collect::<Result<_, Report>>()?;

        let mut root_store = RootCertStore::empty();
        for (id, cert) in &certs {
            root_store
                .add(cert.clone())
                .with_context(|| format!("adding certificate for party {} to root store", id))?;
        }
        let crypto = quinn::rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();

        let client_config = {
            let mut transport_config = TransportConfig::default();
            transport_config.max_idle_timeout(Some(
                IdleTimeout::try_from(Duration::from_secs(60)).unwrap(),
            ));
            // atm clients send keepalive packets
            transport_config.keep_alive_interval(Some(Duration::from_secs(1)));
            let mut client_config =
                ClientConfig::new(Arc::new(QuicClientConfig::try_from(crypto)?));
            client_config.transport_config(Arc::new(transport_config));
            client_config
        };

        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
            std::fs::read(config.key_path).context("reading own key file")?,
        ));
        let server_config =
            quinn::ServerConfig::with_single_cert(vec![certs[&config.my_id].clone()], key)
                .context("creating our server config")?;
        let our_socket_addr = config.bind_addr;

        let mut endpoints = Vec::new();
        let server_endpoint = quinn::Endpoint::server(server_config.clone(), our_socket_addr)?;

        let mut connections = BTreeMap::new();

        for party in config.parties {
            if party.id == config.my_id {
                // skip self
                continue;
            }
            if party.id < config.my_id {
                // connect to party, we are client

                let party_addresses: Vec<SocketAddr> = party
                    .dns_name
                    .to_socket_addrs()
                    .with_context(|| format!("while resolving DNS name for {}", party.dns_name))?
                    .collect();
                if party_addresses.is_empty() {
                    return Err(eyre::eyre!("could not resolve DNS name {}", party.dns_name));
                }
                let party_addr = party_addresses[0];
                let local_client_socket: SocketAddr = match party_addr {
                    SocketAddr::V4(_) => {
                        "0.0.0.0:0".parse().expect("hardcoded IP address is valid")
                    }
                    SocketAddr::V6(_) => "[::]:0".parse().expect("hardcoded IP address is valid"),
                };
                let endpoint = quinn::Endpoint::client(local_client_socket)
                    .with_context(|| format!("creating client endpoint to party {}", party.id))?;
                let conn = endpoint
                    .connect_with(client_config.clone(), party_addr, &party.dns_name.hostname)
                    .with_context(|| {
                        format!("setting up client connection with party {}", party.id)
                    })?
                    .await
                    .with_context(|| format!("connecting as a client to party {}", party.id))?;
                let mut uni = conn.open_uni().await?;
                uni.write_u32(u32::try_from(config.my_id).expect("party id fits into u32"))
                    .await?;
                uni.flush().await?;
                uni.finish()?;
                tracing::trace!(
                    "Conn with id {} from {} to {}",
                    conn.stable_id(),
                    endpoint.local_addr().unwrap(),
                    conn.remote_address(),
                );
                assert!(connections.insert(party.id, conn).is_none());
                endpoints.push(endpoint);
            } else {
                // we are the server, accept a connection
                if let Some(maybe_conn) = server_endpoint.accept().await {
                    let conn = maybe_conn.await?;
                    tracing::trace!(
                        "Conn with id {} from {} to {}",
                        conn.stable_id(),
                        server_endpoint.local_addr().unwrap(),
                        conn.remote_address(),
                    );
                    let mut uni = conn.accept_uni().await?;
                    let other_party_id = uni.read_u32().await?;
                    assert!(connections
                        .insert(
                            usize::try_from(other_party_id).expect("u32 fits into usize"),
                            conn
                        )
                        .is_none());
                } else {
                    return Err(eyre::eyre!(
                        "server endpoint did not accept a connection from party {}",
                        party.id
                    ));
                }
            }
        }
        endpoints.push(server_endpoint);

        Ok(MpcNetworkHandler {
            connections,
            endpoints,
            my_id: config.my_id,
        })
    }

    /// Returns the number of sent and received bytes.
    pub fn get_send_receive(&self, i: usize) -> std::io::Result<(u64, u64)> {
        let conn = self
            .connections
            .get(&i)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such connection"))?;
        let stats = conn.stats();
        Ok((stats.udp_tx.bytes, stats.udp_rx.bytes))
    }

    /// Prints the connection statistics.
    pub fn print_connection_stats(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        for (i, conn) in &self.connections {
            let stats = conn.stats();
            writeln!(
                out,
                "Connection {} stats:\n\tSENT: {} bytes\n\tRECV: {} bytes",
                i, stats.udp_tx.bytes, stats.udp_rx.bytes
            )?;
        }
        Ok(())
    }

    /// Sets up a new [BytesChannel] between each party. The resulting map maps the id of the party to its respective [BytesChannel].
    pub async fn get_byte_channels(
        &self,
    ) -> std::io::Result<HashMap<usize, BytesChannel<RecvStream, SendStream>>> {
        let mut codec = LengthDelimitedCodec::new();
        codec.set_max_frame_length(1_000_000_000);
        self.get_custom_channels(codec).await
    }

    /// Set up a new [Channel] using [BincodeCodec] between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_serde_bincode_channels<M: Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, BincodeCodec<M>>>> {
        let bincodec = BincodeCodec::<M>::new();
        self.get_custom_channels(bincodec).await
    }

    /// Set up a new [Channel] using the provided codec between each party. The resulting map maps the id of the party to its respective [Channel].
    pub async fn get_custom_channels<
        MSend,
        MRecv,
        C: Encoder<MSend, Error = io::Error>
            + Decoder<Item = MRecv, Error = io::Error>
            + 'static
            + Clone,
    >(
        &self,
        codec: C,
    ) -> std::io::Result<HashMap<usize, Channel<RecvStream, SendStream, C>>> {
        let mut channels = HashMap::with_capacity(self.connections.len() - 1);
        for (&id, conn) in self.connections.iter() {
            if id < self.my_id {
                // we are the client, so we are the receiver
                let (mut send_stream, mut recv_stream) = conn.open_bi().await?;
                send_stream.write_u32(self.my_id as u32).await?;
                let their_id = recv_stream.read_u32().await?;
                assert!(their_id == id as u32);
                let conn = Channel::new(recv_stream, send_stream, codec.clone());
                assert!(channels.insert(id, conn).is_none());
            } else {
                // we are the server, so we are the sender
                let (mut send_stream, mut recv_stream) = conn.accept_bi().await?;
                let their_id = recv_stream.read_u32().await?;
                assert!(their_id == id as u32);
                send_stream.write_u32(self.my_id as u32).await?;
                let conn = Channel::new(recv_stream, send_stream, codec.clone());
                assert!(channels.insert(id, conn).is_none());
            }
        }
        Ok(channels)
    }

    /// Shutdown all connections, and call [`quinn::Endpoint::wait_idle`] on all of them
    pub async fn shutdown(self) -> std::io::Result<()> {
        tracing::debug!(
            "party {} shutting down, conns = {:?}",
            self.my_id,
            self.connections.keys()
        );

        for (id, conn) in self.connections.iter() {
            if self.my_id < *id {
                let mut send = conn.open_uni().await?;
                send.write_all(b"done").await?;
            } else {
                let mut recv = conn.accept_uni().await?;
                let mut buffer = vec![0u8; b"done".len()];
                recv.read_exact(&mut buffer).await.map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "failed to recv done msg")
                })?;

                tracing::debug!("party {} closing conn = {id}", self.my_id);

                conn.close(
                    0u32.into(),
                    format!("close from party {}", self.my_id).as_bytes(),
                );
            }
        }
        for endpoint in self.endpoints {
            endpoint.wait_idle().await;
            endpoint.close(VarInt::from_u32(0), &[]);
        }
        Ok(())
    }
}
