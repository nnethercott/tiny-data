use clap::Parser;
use tinydata::client::*;

#[derive(Parser, Debug, Clone)]
pub struct Args {
    //https://stackoverflow.com/questions/74936109/how-to-use-clap-to-take-create-a-vector
    /// Space-delimited list of image classes
    #[arg(short, long, num_args = 1.., value_delimiter = ' ', value_parser)]
    pub topics: Vec<String>,

    /// number of images to download per-class
    #[arg(short, long, default_value_t = 20)]
    pub nsamples: usize,

    /// name of directory to save to
    #[arg(short, long, default_value = "images")]
    pub dir: String,
}

async fn run() {
    let args = Args::parse();
    let tiny_data_client = TinyDataClient::new();
    tiny_data_client
        .run(args.topics, args.nsamples, args.dir)
        .await;
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    run().await;
}
