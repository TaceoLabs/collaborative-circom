//! Rep3 Network
//!
//! This module contains implementation of the rep3 mpc network

use std::{io, sync::Arc};

use crate::RngType;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::{Bytes, BytesMut};
use eyre::{bail, eyre, Report};
use futures::{SinkExt, StreamExt};
use mpc_net::{
    channel::{Channel, ReadChannel, WriteChannel},
    config::NetworkConfig,
    MpcNetworkHandler,
};
use quinn::{RecvStream, SendStream};
use tokio_util::codec::LengthDelimitedCodec;

use super::{
    id::PartyID,
    rngs::{Rep3CorrelatedRng, Rep3Rand, Rep3RandBitComp},
    IoResult,
};
use rand::{Rng, SeedableRng};

// this will be moved later
/// This struct handles networking and rng
pub struct IoContext<N: Rep3Network> {
    /// The party id
    pub id: PartyID,
    /// The rng
    pub rngs: Rep3CorrelatedRng,
    /// The underlying network
    pub network: N,
}

impl<N: Rep3Network> IoContext<N> {
    async fn setup_prf(network: &mut N) -> IoResult<Rep3Rand> {
        let seed1: [u8; crate::SEED_SIZE] = RngType::from_entropy().gen();
        network.send_next(seed1).await?;
        let seed2: [u8; crate::SEED_SIZE] = network.recv_prev().await?;

        Ok(Rep3Rand::new(seed1, seed2))
    }

    async fn setup_bitcomp(
        network: &mut N,
        rands: &mut Rep3Rand,
    ) -> IoResult<(Rep3RandBitComp, Rep3RandBitComp)> {
        let (k1a, k1c) = rands.random_seeds();
        let (k2a, k2c) = rands.random_seeds();

        match network.get_id() {
            PartyID::ID0 => {
                network.send_next(k1c).await?;
                let k2b: [u8; crate::SEED_SIZE] = network.recv_prev().await?;
                let bitcomp1 = Rep3RandBitComp::new_2keys(k1a, k1c);
                let bitcomp2 = Rep3RandBitComp::new_3keys(k2a, k2b, k2c);
                Ok((bitcomp1, bitcomp2))
            }
            PartyID::ID1 => {
                network.send_next((k1c, k2c)).await?;
                let k1b: [u8; crate::SEED_SIZE] = network.recv_prev().await?;
                let bitcomp1 = Rep3RandBitComp::new_3keys(k1a, k1b, k1c);
                let bitcomp2 = Rep3RandBitComp::new_2keys(k2a, k2c);
                Ok((bitcomp1, bitcomp2))
            }
            PartyID::ID2 => {
                network.send_next(k2c).await?;
                let (k1b, k2b): ([u8; crate::SEED_SIZE], [u8; crate::SEED_SIZE]) =
                    network.recv_prev().await?;
                let bitcomp1 = Rep3RandBitComp::new_3keys(k1a, k1b, k1c);
                let bitcomp2 = Rep3RandBitComp::new_3keys(k2a, k2b, k2c);
                Ok((bitcomp1, bitcomp2))
            }
        }
    }

    /// Construct  a new [`IoContext`] with the given network
    pub async fn init(mut network: N) -> IoResult<Self> {
        let mut rand = Self::setup_prf(&mut network).await?;
        let bitcomps = Self::setup_bitcomp(&mut network, &mut rand).await?;
        let rngs = Rep3CorrelatedRng::new(rand, bitcomps.0, bitcomps.1);

        Ok(Self {
            id: network.get_id(), //shorthand access
            network,
            rngs,
        })
    }

    /// Generate two random field elements
    pub fn random_fes<F: PrimeField>(&mut self) -> (F, F) {
        self.rngs.rand.random_fes()
    }

    /// Cronstruct a fork of the [`IoContext`]. This fork can be used concurrently with its parent.
    pub async fn fork(&mut self) -> IoResult<Self> {
        let network = self.network.fork().await?;
        let rngs = self.rngs.fork();
        let id = self.id;

        Ok(Self { id, rngs, network })
    }
}

#[allow(async_fn_in_trait)]
/// This trait defines the network interface for the REP3 protocol.
pub trait Rep3Network: Send {
    /// Returns the id of the party. The id is in the range 0 <= id < 3
    fn get_id(&self) -> PartyID;

    /// Sends `data` to the next party and receives from the previous party. Use this whenever
    /// possible in contrast to calling [`Self::send_next()`] and [`Self::recv_prev()`] sequential. This method
    /// executes send/receive concurrently.
    async fn reshare<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<F> {
        let mut res = self.reshare_many(&[data]).await?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            //we checked that there is really one element
            Ok(res.pop().unwrap())
        }
    }

    /// Perform multiple reshares with one networking round
    async fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<Vec<F>>;

    /// Broadcast data to the other two parties and receive data from them
    async fn broadcast<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<(F, F)> {
        let (mut prev, mut next) = self.broadcast_many(&[data]).await?;
        if prev.len() != 1 || next.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            //we checked that there is really one element
            let prev = prev.pop().unwrap();
            let next = next.pop().unwrap();
            Ok((prev, next))
        }
    }

    /// Broadcast data to the other two parties and receive data from them
    async fn broadcast_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<(Vec<F>, Vec<F>)>;

    /// Sends data to the target party. This function has a default implementation for calling [Rep3Network::send_many].
    async fn send<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: F,
    ) -> std::io::Result<()> {
        self.send_many(target, &[data]).await
    }

    /// Sends a vector of data to the target party.
    async fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()>;

    /// Sends data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send] with the next_id.
    async fn send_next<F: CanonicalSerialize>(&mut self, data: F) -> std::io::Result<()> {
        self.send(self.get_id().next_id(), data).await
    }

    /// Sends a vector data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send_many] with the next_id.
    async fn send_next_many<F: CanonicalSerialize>(&mut self, data: &[F]) -> std::io::Result<()> {
        self.send_many(self.get_id().next_id(), data).await
    }

    /// Receives data from the party with the given id. This function has a default implementation for calling [Rep3Network::recv_many] and checking for the correct length of 1.
    async fn recv<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<F> {
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
    async fn recv_many<F: CanonicalDeserialize>(
        &mut self,
        from: PartyID,
    ) -> std::io::Result<Vec<F>>;

    /// Receives data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv] with the prev_id.
    async fn recv_prev<F: CanonicalDeserialize>(&mut self) -> std::io::Result<F> {
        self.recv(self.get_id().prev_id()).await
    }

    /// Receives a vector of data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv_many] with the prev_id.
    async fn recv_prev_many<F: CanonicalDeserialize>(&mut self) -> std::io::Result<Vec<F>> {
        self.recv_many(self.get_id().prev_id()).await
    }

    /// Fork the network into two separate instances with their own connections
    async fn fork(&mut self) -> std::io::Result<Self>
    where
        Self: Sized;

    /// Shutdown the network
    async fn shutdown(self) -> std::io::Result<()>;
}

// TODO make generic over codec?
/// This struct can be used to facilitate network communication for the REP3 MPC protocol.
#[derive(Debug)]
pub struct Rep3MpcNet {
    pub(crate) id: PartyID,
    pub(crate) net_handler: Arc<MpcNetworkHandler>,
    pub(crate) chan_next: Channel<RecvStream, SendStream, LengthDelimitedCodec>,
    pub(crate) chan_prev: Channel<RecvStream, SendStream, LengthDelimitedCodec>,
}

impl Rep3MpcNet {
    /// Takes a [NetworkConfig] struct and constructs the network interface. The network needs to contain exactly 3 parties with ids 0, 1, and 2.
    pub async fn new(config: NetworkConfig) -> Result<Self, Report> {
        if config.parties.len() != 3 {
            bail!("REP3 protocol requires exactly 3 parties")
        }
        let id = PartyID::try_from(config.my_id)?;
        let net_handler = Arc::new(MpcNetworkHandler::establish(config).await?);
        let mut channels = net_handler.get_byte_channels().await?;
        let chan_next = channels
            .remove(&id.next_id().into())
            .ok_or(eyre!("no next channel found"))?;
        let chan_prev = channels
            .remove(&id.prev_id().into())
            .ok_or(eyre!("no prev channel found"))?;
        if !channels.is_empty() {
            bail!("unexpected channels found")
        }

        Ok(Self {
            id,
            net_handler,
            chan_next,
            chan_prev,
        })
    }

    async fn send_raw<F: CanonicalSerialize>(
        data: &[F],
        write: &mut WriteChannel<SendStream, LengthDelimitedCodec>,
    ) -> io::Result<()> {
        let mut bytes = Vec::with_capacity(data.serialized_size(ark_serialize::Compress::No));
        data.serialize_uncompressed(&mut bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        write.send(Bytes::from(bytes)).await?;
        Ok(())
    }

    async fn recv_raw<F: CanonicalDeserialize>(
        read: &mut ReadChannel<RecvStream, LengthDelimitedCodec>,
    ) -> io::Result<Vec<F>> {
        let bytes = read.next().await.expect("recv none")?;
        let res = Vec::<F>::deserialize_uncompressed(&bytes[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(res)
    }

    async fn reshare_send_many<F: CanonicalSerialize>(
        data: &[F],
        chan_next: &mut Channel<RecvStream, SendStream, LengthDelimitedCodec>,
    ) -> io::Result<()> {
        let (write, _) = chan_next.inner_ref();
        Self::send_raw(data, write).await
    }

    async fn reshare_recv_many<F: CanonicalDeserialize>(
        chan_prev: &mut Channel<RecvStream, SendStream, LengthDelimitedCodec>,
    ) -> io::Result<Vec<F>> {
        let (_, read) = chan_prev.inner_ref();
        Self::recv_raw(read).await
    }

    /// Sends bytes over the network to the target party.
    pub async fn send_bytes(&mut self, target: PartyID, data: Bytes) -> std::io::Result<()> {
        if target == self.id.next_id() {
            self.chan_next.send(data).await?;
            Ok(())
        } else if target == self.id.prev_id() {
            self.chan_prev.send(data).await?;
            Ok(())
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot send to self",
            ));
        }
    }

    /// Receives bytes over the network from the party with the given id.
    pub async fn recv_bytes(&mut self, from: PartyID) -> std::io::Result<BytesMut> {
        // TODO remove expect
        if from == self.id.prev_id() {
            self.chan_prev.next().await.expect("recv none")
        } else if from == self.id.next_id() {
            self.chan_next.next().await.expect("recv none")
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Cannot recv from self",
            ));
        }
    }
}

impl Rep3Network for Rep3MpcNet {
    fn get_id(&self) -> PartyID {
        self.id
    }

    async fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<Vec<F>> {
        let (send, recv) = tokio::join!(
            Self::reshare_send_many(data, &mut self.chan_next),
            Self::reshare_recv_many(&mut self.chan_prev)
        );
        send?;
        recv
    }

    async fn broadcast_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<(Vec<F>, Vec<F>)> {
        let (send_next, recv_next) = self.chan_next.inner_ref();
        let (send_prev, recv_prev) = self.chan_prev.inner_ref();
        let (a, b, c, d) = tokio::join!(
            Self::send_raw(data, send_next),
            Self::send_raw(data, send_prev),
            Self::recv_raw::<F>(recv_prev),
            Self::recv_raw::<F>(recv_next),
        );
        a?;
        b?;
        Ok((c?, d?))
    }

    async fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()> {
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut ser_data = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut ser_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        self.send_bytes(target, Bytes::from(ser_data)).await
    }

    async fn recv_many<F: CanonicalDeserialize>(
        &mut self,
        from: PartyID,
    ) -> std::io::Result<Vec<F>> {
        let data = self.recv_bytes(from).await?;

        let res = Vec::<F>::deserialize_uncompressed(&data[..])
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(res)
    }

    async fn fork(&mut self) -> std::io::Result<Self> {
        let id = self.id;
        let net_handler = Arc::clone(&self.net_handler);
        let mut channels = net_handler.get_byte_channels().await?;

        Ok(Self {
            id,
            net_handler,
            chan_next: channels.remove(&id.next_id().into()).unwrap(),
            chan_prev: channels.remove(&id.prev_id().into()).unwrap(),
        })
    }

    async fn shutdown(self) -> std::io::Result<()> {
        if let Some(net_handler) = Arc::into_inner(self.net_handler) {
            net_handler.shutdown().await
        } else {
            Ok(())
        }
    }
}
