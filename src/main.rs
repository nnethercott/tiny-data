use clap::Parser;
use tiny_data::client::*;

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let args = Args::parse();

    let mut tiny_data_client = TinyDataClient::new(args);
    tiny_data_client.run().await;
}
