use bytes::Bytes;
use tokio::sync::mpsc::UnboundedReceiver;

pub mod rep3_network;
pub mod shamir_network;

#[derive(Debug)]
pub enum Msg {
    Data(Bytes),
    Recv(UnboundedReceiver<Msg>),
}

impl Msg {
    fn into_recv(self) -> Option<UnboundedReceiver<Msg>> {
        if let Msg::Recv(x) = self {
            Some(x)
        } else {
            None
        }
    }

    fn into_data(self) -> Option<Bytes> {
        if let Msg::Data(x) = self {
            Some(x)
        } else {
            None
        }
    }
}
