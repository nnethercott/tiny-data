use clap::Parser;
use std::time::Instant;
use tiny_data::client::*;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), reqwest::Error> {
    let args = Args::parse();

    let now = Instant::now();

    let mut tiny_data_client = TinyDataClient::new(args);
    tiny_data_client.run().await;

    println!("{:?}", now.elapsed());

    Ok(())
}
