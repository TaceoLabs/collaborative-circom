use super::shamir_network::PartyTestNetwork as ShamirPartyTestNetwork;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::Bytes;
use mpc_core::protocols::{
    bridges::network::RepToShamirNetwork,
    rep3::{id::PartyID, network::Rep3Network},
};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::Msg;

pub struct Rep3TestNetwork {
    p1_p2_sender: UnboundedSender<Msg>,
    p1_p3_sender: UnboundedSender<Msg>,
    p2_p3_sender: UnboundedSender<Msg>,
    p2_p1_sender: UnboundedSender<Msg>,
    p3_p1_sender: UnboundedSender<Msg>,
    p3_p2_sender: UnboundedSender<Msg>,
    p1_p2_receiver: UnboundedReceiver<Msg>,
    p1_p3_receiver: UnboundedReceiver<Msg>,
    p2_p3_receiver: UnboundedReceiver<Msg>,
    p2_p1_receiver: UnboundedReceiver<Msg>,
    p3_p1_receiver: UnboundedReceiver<Msg>,
    p3_p2_receiver: UnboundedReceiver<Msg>,
}

impl Default for Rep3TestNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl Rep3TestNetwork {
    pub fn new() -> Self {
        // AT Most 1 message is buffered before they are read so this should be fine
        let p1_p2 = mpsc::unbounded_channel();
        let p1_p3 = mpsc::unbounded_channel();
        let p2_p3 = mpsc::unbounded_channel();
        let p2_p1 = mpsc::unbounded_channel();
        let p3_p1 = mpsc::unbounded_channel();
        let p3_p2 = mpsc::unbounded_channel();

        Self {
            p1_p2_sender: p1_p2.0,
            p1_p3_sender: p1_p3.0,
            p2_p1_sender: p2_p1.0,
            p2_p3_sender: p2_p3.0,
            p3_p1_sender: p3_p1.0,
            p3_p2_sender: p3_p2.0,
            p1_p2_receiver: p1_p2.1,
            p1_p3_receiver: p1_p3.1,
            p2_p1_receiver: p2_p1.1,
            p2_p3_receiver: p2_p3.1,
            p3_p1_receiver: p3_p1.1,
            p3_p2_receiver: p3_p2.1,
        }
    }

    pub fn get_party_networks(self) -> [PartyTestNetwork; 3] {
        let party1 = PartyTestNetwork {
            id: PartyID::ID0,
            send_prev: self.p1_p3_sender,
            recv_prev: self.p3_p1_receiver,
            send_next: self.p1_p2_sender,
            recv_next: self.p2_p1_receiver,
            _stats: [0; 4],
        };

        let party2 = PartyTestNetwork {
            id: PartyID::ID1,
            send_prev: self.p2_p1_sender,
            recv_prev: self.p1_p2_receiver,
            send_next: self.p2_p3_sender,
            recv_next: self.p3_p2_receiver,
            _stats: [0; 4],
        };

        let party3 = PartyTestNetwork {
            id: PartyID::ID2,
            send_prev: self.p3_p2_sender,
            recv_prev: self.p2_p3_receiver,
            send_next: self.p3_p1_sender,
            recv_next: self.p1_p3_receiver,
            _stats: [0; 4],
        };

        [party1, party2, party3]
    }
}

#[derive(Debug)]
pub struct PartyTestNetwork {
    pub id: PartyID,
    pub send_prev: UnboundedSender<Msg>,
    pub send_next: UnboundedSender<Msg>,
    pub recv_prev: UnboundedReceiver<Msg>,
    pub recv_next: UnboundedReceiver<Msg>,
    pub _stats: [usize; 4], // [sent_prev, sent_next, recv_prev, recv_next]
}

impl Rep3Network for PartyTestNetwork {
    fn get_id(&self) -> PartyID {
        self.id
    }

    async fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<Vec<F>> {
        self.send_next_many(data).await?;
        self.recv_prev_many().await
    }

    async fn broadcast<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<(F, F)> {
        let data = [data];
        self.send_many(self.id.next_id(), &data).await?;
        self.send_many(self.id.prev_id(), &data).await?;
        let mut prev = self.recv_many(self.id.prev_id()).await?;
        let mut next = self.recv_many(self.id.next_id()).await?;
        if next.len() != 1 || prev.len() != 1 {
            panic!("got more than one from next or prev");
        }
        Ok((prev.pop().unwrap(), next.pop().unwrap()))
    }

    async fn broadcast_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<(Vec<F>, Vec<F>)> {
        self.send_many(self.id.next_id(), data).await?;
        self.send_many(self.id.prev_id(), data).await?;
        let prev = self.recv_many(self.id.prev_id()).await?;
        let next = self.recv_many(self.id.next_id()).await?;
        Ok((prev, next))
    }

    async fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()> {
        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut to_send = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut to_send).unwrap();
        if self.id.next_id() == target {
            self.send_next
                .send(Msg::Data(Bytes::from(to_send)))
                .expect("can send to next")
        } else if self.id.prev_id() == target {
            self.send_prev
                .send(Msg::Data(Bytes::from(to_send)))
                .expect("can send to next");
        } else {
            panic!("You want to send to yourself?")
        }
        Ok(())
    }

    async fn recv_many<F: CanonicalDeserialize>(
        &mut self,
        from: PartyID,
    ) -> std::io::Result<Vec<F>> {
        if self.id.next_id() == from {
            let data = Vec::from(self.recv_next.recv().await.unwrap().into_data().unwrap());
            Ok(Vec::<F>::deserialize_uncompressed(data.as_slice()).unwrap())
        } else if self.id.prev_id() == from {
            let data = Vec::from(self.recv_prev.recv().await.unwrap().into_data().unwrap());
            Ok(Vec::<F>::deserialize_uncompressed(data.as_slice()).unwrap())
        } else {
            panic!("You want to read from yourself?")
        }
    }

    async fn fork(&mut self) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let ch_prev = mpsc::unbounded_channel();
        let ch_next = mpsc::unbounded_channel();

        self.send_next.send(Msg::Recv(ch_next.1)).unwrap();
        self.send_prev.send(Msg::Recv(ch_prev.1)).unwrap();

        let recv_prev = self.recv_prev.recv().await.unwrap().into_recv().unwrap();
        let recv_next = self.recv_next.recv().await.unwrap().into_recv().unwrap();

        let id = self.id;

        Ok(Self {
            id,
            send_prev: ch_prev.0,
            send_next: ch_next.0,
            recv_prev,
            recv_next,
            _stats: [0; 4],
        })
    }

    async fn shutdown(self) -> std::io::Result<()> {
        Ok(())
    }
}

impl RepToShamirNetwork<ShamirPartyTestNetwork> for PartyTestNetwork {
    fn to_shamir_net(self) -> ShamirPartyTestNetwork {
        let Self {
            id,
            send_prev,
            send_next,
            recv_prev,
            recv_next,
            _stats,
        } = self;

        let mut send = Vec::with_capacity(2);
        let mut recv = Vec::with_capacity(2);

        match id {
            PartyID::ID0 => {
                send.push(send_next);
                send.push(send_prev);
                recv.push(recv_next);
                recv.push(recv_prev);
            }
            PartyID::ID1 => {
                send.push(send_prev);
                send.push(send_next);
                recv.push(recv_prev);
                recv.push(recv_next);
            }
            PartyID::ID2 => {
                send.push(send_next);
                send.push(send_prev);
                recv.push(recv_next);
                recv.push(recv_prev);
            }
        }

        ShamirPartyTestNetwork {
            id: id.into(),
            num_parties: 3,
            send,
            recv,
        }
    }
}
