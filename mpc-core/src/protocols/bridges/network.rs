//! # Network bridges
//!
//! This module contains code to translate networks used for different MPC protocols into each other.

use std::collections::HashMap;

use crate::protocols::{
    rep3::network::{Rep3MpcNet, Rep3Network},
    shamir::network::{ShamirMpcNet, ShamirNetwork},
};

/// This trait represents the possibility to transform a network implementation of the [Rep3Network] trait (used for 3-party replicated secret sharing) into a 3-party network implementation of the [ShamirNetwork] trait (used for 3-party Shamir secret sharing).
pub trait RepToShamirNetwork<N: ShamirNetwork>: Rep3Network {
    /// Translates the network into a 3-party Shamir network.
    fn to_shamir_net(self) -> N;
}

impl RepToShamirNetwork<ShamirMpcNet> for Rep3MpcNet {
    fn to_shamir_net(self) -> ShamirMpcNet {
        let Self {
            id,
            net_handler,
            chan_next,
            chan_prev,
        } = self;

        let mut channels = HashMap::with_capacity(2);
        channels.insert(id.next_id().into(), chan_next);
        channels.insert(id.prev_id().into(), chan_prev);

        ShamirMpcNet {
            id: id.into(),
            num_parties: 3,
            net_handler,
            channels,
        }
    }
}
