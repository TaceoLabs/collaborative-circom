//! Shamir Network
//!
//! This module contains the trait for specifying a network interface for the Shamir MPC protocol. It also contains an implementation of the trait using the [mpc_net] crate.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::{Bytes, BytesMut};
use eyre::{bail, Report};
use futures::{SinkExt, StreamExt};
use mpc_net::{channel::Channel, config::NetworkConfig, MpcNetworkHandler};
use quinn::{RecvStream, SendStream};
use std::{collections::HashMap, sync::Arc};
use tokio_util::codec::LengthDelimitedCodec;

#[allow(async_fn_in_trait)]
/// This trait defines the network interface for the Shamir protocol.
pub trait ShamirNetwork: Send {
    /// Returns the id of the party. The id is in the range 0 <= id < num_parties
    fn get_id(&self) -> usize;

    /// Returns the number of parties participating in the MPC protocol.
    fn get_num_parties(&self) -> usize;

    /// Sends data to the target party. This function has a default implementation for calling [ShamirNetwork::send_many].
    async fn send<F: CanonicalSerialize>(&mut self, target: usize, data: F) -> std::io::Result<()> {
        self.send_many(target, &[data]).await
    }

    /// Sends a vector of data to the target party.
    async fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: usize,
        data: &[F],
    ) -> std::io::Result<()>;

    /// Receives data from the party with the given id. This function has a default implementation for calling [ShamirNetwork::recv_many] and checking for the correct length of 1.
    async fn recv<F: CanonicalDeserialize>(&mut self, from: usize) -> std::io::Result<F> {
        let mut res = self.recv_many(from).await?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            Ok(res.pop().unwrap())
        }
    }

    /// Receives a vector of data from the party with the given id.
    async fn recv_many<F: CanonicalDeserialize>(&mut self, from: usize) -> std::io::Result<Vec<F>>;

    /// Sends data to all parties and receives data from all other parties. The result is a vector where the data from party i is at index i, including my own data.
    async fn broadcast<F: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        data: F,
    ) -> std::io::Result<Vec<F>>;

    /// Sends data to the next num - 1 parties and receives from the previous num -1 parties. Thus, the result is a vector of length num, where the data from party my_id + num_partes - i mod num_parties is at index i, including my own data.
    async fn broadcast_next<F: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        data: F,
        num: usize,
    ) -> std::io::Result<Vec<F>>;

    /// Sends and receives to and from each party. Data must be of shape num_parties x n. The element that is "sent" to yourself is passed back directly.
    async fn send_and_recv_each_many<
        F: CanonicalSerialize + CanonicalDeserialize + Clone + Send + 'static,
    >(
        &mut self,
        data: Vec<Vec<F>>,
    ) -> std::io::Result<Vec<Vec<F>>>;

    /// Fork the network into two separate instances with their own connections
    async fn fork(&mut self) -> std::io::Result<Self>
    where
        Self: Sized;

    /// Shutdown the network
    async fn shutdown(self) -> std::io::Result<()>;
}

/// This struct can be used to facilitate network communication for the Shamir MPC protocol.
pub struct ShamirMpcNet {
    pub(crate) id: usize, // 0 <= id < num_parties
    pub(crate) num_parties: usize,
    pub(crate) net_handler: Arc<MpcNetworkHandler>,
    pub(crate) channels: HashMap<usize, Channel<RecvStream, SendStream, LengthDelimitedCodec>>,
}

impl ShamirMpcNet {
    /// Takes a [NetworkConfig] struct and constructs the network interface. The network needs to contain at least 3 parties and all ids need to be in the range of 0 <= id < num_parties.
    pub async fn new(config: NetworkConfig) -> Result<Self, Report> {
        let num_parties = config.parties.len();

        if config.parties.len() <= 2 {
            bail!("Shamir protocol requires at least 3 parties")
        }
        let id = config.my_id;
        if id >= num_parties {
            bail!("Invalid party id={} for {} parties", id, num_parties)
        }

        let net_handler = MpcNetworkHandler::establish(config).await?;
        let channels = net_handler.get_byte_channels().await?;

        Ok(Self {
            id,
            num_parties,
            net_handler: Arc::new(net_handler),
            channels,
        })
    }

    /// Sends bytes over the network to the target party.
    pub async fn send_bytes(&mut self, target: usize, data: Bytes) -> std::io::Result<()> {
        if let Some(chan) = self.channels.get_mut(&target) {
            chan.send(data).await?;
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("No channel found for party id={}", target),
            ))
        }
    }

    /// Receives bytes over the network from the party with the given id.
    pub async fn recv_bytes(&mut self, from: usize) -> std::io::Result<BytesMut> {
        let data = if let Some(chan) = self.channels.get_mut(&from) {
            chan.next().await.ok_or(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Received None",
            ))?
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("No channel found for party id={}", from),
            ));
        };

        let data = data.map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::BrokenPipe, "receive channel end died")
        })?;
        Ok(data)
    }

    pub(crate) fn _id(&self) -> usize {
        self.id
    }
}

impl ShamirNetwork for ShamirMpcNet {
    fn get_id(&self) -> usize {
        self.id
    }

    fn get_num_parties(&self) -> usize {
        self.num_parties
    }

    async fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: usize,
        data: &[F],
    ) -> std::io::Result<()> {
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        self.send_bytes(target, Bytes::from(ser_data)).await
    }

    async fn recv_many<F: CanonicalDeserialize>(&mut self, from: usize) -> std::io::Result<Vec<F>> {
        let data = self.recv_bytes(from).await?;

        let res = Vec::<F>::deserialize_uncompressed(&data[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(res)
    }

    async fn broadcast<F: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        data: F,
    ) -> std::io::Result<Vec<F>> {
        // Serialize
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.to_owned()
            .serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let send_data = Bytes::from(ser_data);

        // Send
        for other_id in 0..self.num_parties {
            if other_id != self.id {
                self.send_bytes(other_id, send_data.to_owned()).await?;
            }
        }

        // Receive
        let mut res = Vec::with_capacity(self.num_parties);
        for other_id in 0..self.num_parties {
            if other_id != self.id {
                let data = self.recv_bytes(other_id).await?;
                let deser = F::deserialize_uncompressed(&data[..])
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                res.push(deser);
            } else {
                res.push(data.to_owned());
            }
        }

        Ok(res)
    }

    async fn broadcast_next<F: CanonicalSerialize + CanonicalDeserialize + Clone>(
        &mut self,
        data: F,
        num: usize,
    ) -> std::io::Result<Vec<F>> {
        // Serialize
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.to_owned()
            .serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let send_data = Bytes::from(ser_data);

        // Send
        for s in 1..num {
            let other_id = (self.id + s) % self.num_parties;
            // if other_id != self.id {
            self.send_bytes(other_id, send_data.to_owned()).await?;
            // }
        }

        // Receive
        let mut res = Vec::with_capacity(num);
        res.push(data);
        for r in 1..num {
            let other_id = (self.id + self.num_parties - r) % self.num_parties;
            let data = self.recv_bytes(other_id).await?;
            let deser = F::deserialize_uncompressed(&data[..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            res.push(deser);
        }

        Ok(res)
    }

    async fn fork(&mut self) -> std::io::Result<Self> {
        let id = self.id;
        let num_parties = self.num_parties;
        let net_handler = Arc::clone(&self.net_handler);
        let channels = net_handler.get_byte_channels().await?;

        Ok(Self {
            id,
            num_parties,
            net_handler,
            channels,
        })
    }

    async fn shutdown(self) -> std::io::Result<()> {
        if let Some(net_handler) = Arc::into_inner(self.net_handler) {
            net_handler.shutdown().await
        } else {
            Ok(())
        }
    }

    async fn send_and_recv_each_many<
        F: CanonicalSerialize + CanonicalDeserialize + Clone + Send + 'static,
    >(
        &mut self,
        data: Vec<Vec<F>>,
    ) -> std::io::Result<Vec<Vec<F>>> {
        debug_assert_eq!(data.len(), self.num_parties);
        let mut res = Vec::with_capacity(data.len());

        // move channels and data out of self and input so we can move them into tokio::spawn
        let futures = (0..self.num_parties)
            .zip(data)
            .map(|(id, data)| {
                let chan = self.channels.remove(&id);
                tokio::spawn(async move {
                    if let Some(chan) = chan {
                        let (mut write, mut read) = chan.split();
                        let (_, recv) = tokio::try_join!(
                            async {
                                let size = data.serialized_size(ark_serialize::Compress::No);
                                let mut ser_data = Vec::with_capacity(size);
                                data.serialize_uncompressed(&mut ser_data).map_err(|e| {
                                    std::io::Error::new(std::io::ErrorKind::InvalidInput, e)
                                })?;
                                write.send(ser_data.into()).await
                            },
                            async {
                                let data = read.next().await.ok_or(std::io::Error::new(
                                    std::io::ErrorKind::Other,
                                    "Received None",
                                ))??;
                                let res =
                                    Vec::<F>::deserialize_uncompressed(&data[..]).map_err(|e| {
                                        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                                    })?;
                                Ok(res)
                            }
                        )?;
                        Ok::<_, std::io::Error>((Some(Channel::join(write, read)), recv))
                    } else {
                        Ok((None, data))
                    }
                })
            })
            .collect::<Vec<_>>();

        // collect results of futures and move channels back into self.channels
        for (id, e) in futures::future::try_join_all(futures)
            .await?
            .into_iter()
            .enumerate()
        {
            let (chan, recv) = e?;
            // only insert chan were there was one
            if let Some(c) = chan {
                self.channels.insert(id, c);
            }
            res.push(recv);
        }

        Ok(res)
    }
}
