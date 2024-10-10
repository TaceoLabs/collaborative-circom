use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bytes::Bytes;
use mpc_core::protocols::shamir::network::ShamirNetwork;
use std::{cmp::Ordering, collections::HashMap};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::Msg;

pub struct ShamirTestNetwork {
    num_parties: usize,
    sender: HashMap<(usize, usize), UnboundedSender<Msg>>,
    receiver: HashMap<(usize, usize), UnboundedReceiver<Msg>>,
}

impl ShamirTestNetwork {
    pub fn new(num_parties: usize) -> Self {
        // AT Most 1 message is buffered before they are read so this should be fine
        let mut sender = HashMap::with_capacity(num_parties * (num_parties - 1));
        let mut receiver = HashMap::with_capacity(num_parties * (num_parties - 1));

        for sender_id in 0..num_parties {
            for mut receiver_id in 0..num_parties - 1 {
                if receiver_id >= sender_id {
                    receiver_id += 1;
                }
                let (s, r) = mpsc::unbounded_channel();
                sender.insert((sender_id, receiver_id), s);
                receiver.insert((sender_id, receiver_id), r);
            }
        }

        Self {
            num_parties,
            sender,
            receiver,
        }
    }

    pub fn get_party_networks(mut self) -> Vec<PartyTestNetwork> {
        let mut res = Vec::with_capacity(self.num_parties);

        for partyid in 0..self.num_parties {
            let mut send = Vec::with_capacity(self.num_parties - 1);
            let mut recv = Vec::with_capacity(self.num_parties - 1);

            for mut other_party in 0..self.num_parties - 1 {
                if other_party >= partyid {
                    other_party += 1;
                }

                let s = self.sender.remove(&(partyid, other_party)).unwrap();
                let r = self.receiver.remove(&(other_party, partyid)).unwrap();

                send.push(s);
                recv.push(r);
            }
            let network = PartyTestNetwork {
                id: partyid,
                num_parties: self.num_parties,
                send,
                recv,
            };
            res.push(network);
        }

        assert!(self.sender.is_empty());
        assert!(self.receiver.is_empty());

        res
    }
}

#[derive(Debug)]
pub struct PartyTestNetwork {
    pub id: usize,
    pub num_parties: usize,
    pub send: Vec<UnboundedSender<Msg>>,
    pub recv: Vec<UnboundedReceiver<Msg>>,
}

impl ShamirNetwork for PartyTestNetwork {
    fn get_id(&self) -> usize {
        self.id
    }

    fn get_num_parties(&self) -> usize {
        self.num_parties
    }

    async fn send<F: CanonicalSerialize>(&mut self, target: usize, data: F) -> std::io::Result<()> {
        self.send_many(target, &[data]).await
    }

    async fn send_many<F: CanonicalSerialize>(
        &mut self,
        mut target: usize,
        data: &[F],
    ) -> std::io::Result<()> {
        if target >= self.num_parties || target == self.id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("No channel found for party id={}", target),
            ));
        }
        if target > self.id {
            // to get index for the Vec
            target -= 1;
        }

        let size = data.serialized_size(ark_serialize::Compress::No);
        let mut to_send = Vec::with_capacity(size);
        data.serialize_uncompressed(&mut to_send).unwrap();

        self.send[target]
            .send(Msg::Data(Bytes::from(to_send)))
            .expect("can send");

        Ok(())
    }

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

    async fn recv_many<F: CanonicalDeserialize>(
        &mut self,
        mut from: usize,
    ) -> std::io::Result<Vec<F>> {
        if from >= self.num_parties || from == self.id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("No channel found for party id={}", from),
            ));
        }
        if from > self.id {
            // to get index for the Vec
            from -= 1;
        }
        let data = Vec::from(self.recv[from].recv().await.unwrap().into_data().unwrap());
        Ok(Vec::<F>::deserialize_uncompressed(data.as_slice()).unwrap())
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
        for send in self.send.iter_mut() {
            send.send(Msg::Data(send_data.to_owned()))
                .expect("can send");
        }

        // Receive
        let mut res = Vec::with_capacity(self.num_parties);
        for (other_id, recv) in self.recv.iter_mut().enumerate() {
            if other_id == self.id {
                // Put that in the middle
                res.push(data.to_owned());
            }

            let data = Vec::from(recv.recv().await.unwrap().into_data().unwrap());
            res.push(F::deserialize_uncompressed(data.as_slice()).unwrap());
        }
        if self.id == self.num_parties - 1 {
            // Put that at the end
            res.push(data.to_owned());
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
            let mut other_id = (self.id + s) % self.num_parties;
            match other_id.cmp(&self.id) {
                Ordering::Greater => other_id -= 1,
                Ordering::Less => {}
                Ordering::Equal => continue,
            }
            self.send[other_id]
                .send(Msg::Data(send_data.to_owned()))
                .expect("can send");
        }

        // Receive
        let mut res = Vec::with_capacity(num);
        res.push(data.to_owned());
        for r in 1..num {
            let mut other_id = (self.id + self.num_parties - r) % self.num_parties;
            match other_id.cmp(&self.id) {
                Ordering::Greater => other_id -= 1,
                Ordering::Less => {}
                Ordering::Equal => {
                    res.push(data.to_owned());
                    continue;
                }
            }
            let data = Vec::from(
                self.recv[other_id]
                    .recv()
                    .await
                    .unwrap()
                    .into_data()
                    .unwrap(),
            );
            res.push(F::deserialize_uncompressed(data.as_slice()).unwrap());
        }

        Ok(res)
    }

    async fn fork(&mut self) -> std::io::Result<Self>
    where
        Self: Sized,
    {
        let mut send = Vec::with_capacity(self.num_parties - 1);
        for sender in self.send.iter() {
            let (s, r) = mpsc::unbounded_channel();
            sender.send(Msg::Recv(r)).unwrap();
            send.push(s);
        }

        let mut recv = Vec::with_capacity(self.num_parties - 1);
        for recveiver in self.recv.iter_mut() {
            let r = recveiver.recv().await.unwrap().into_recv().unwrap();
            recv.push(r);
        }

        let id = self.id;
        let num_parties = self.num_parties;

        Ok(Self {
            id,
            num_parties,
            send,
            recv,
        })
    }

    async fn shutdown(self) -> std::io::Result<()> {
        // we do not care about gracefull shutdown
        Ok(())
    }

    async fn send_and_recv_each_many<
        F: CanonicalSerialize + CanonicalDeserialize + Clone + Send + 'static,
    >(
        &mut self,
        data: Vec<Vec<F>>,
    ) -> std::io::Result<Vec<Vec<F>>> {
        debug_assert_eq!(data.len(), self.num_parties);
        let mut res = vec![Vec::new(); self.num_parties];
        for id in 0..self.num_parties {
            if id == self.id {
                res[id] = data[id].clone();
            } else {
                let corr_id = if id > self.id { id - 1 } else { id };
                let data = data[id].clone();
                // send
                let size = data.serialized_size(ark_serialize::Compress::No);
                let mut to_send = Vec::with_capacity(size);
                data.serialize_uncompressed(&mut to_send).unwrap();

                self.send[corr_id]
                    .send(Msg::Data(Bytes::from(to_send)))
                    .expect("can send");

                // receive
                let bytes = Vec::from(
                    self.recv[corr_id]
                        .recv()
                        .await
                        .unwrap()
                        .into_data()
                        .unwrap(),
                );
                let v = Vec::<F>::deserialize_uncompressed(bytes.as_slice()).unwrap();
                res[id] = v;
            }
        }

        Ok(res)
    }
}
