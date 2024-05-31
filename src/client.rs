use clap::Parser;
use futures::future::join_all;
use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use std::fs::create_dir_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinHandle;

use crate::download::*;
use crate::fetch::*;

#[derive(Parser, Debug, Clone)]
pub struct Args {
    //https://stackoverflow.com/questions/74936109/how-to-use-clap-to-take-create-a-vector
    /// Space-delimited list of image classes
    // #[arg(short, long, num_args = 1.., value_delimiter = ' ', value_parser)]
    #[arg(short, long, num_args = 1.., use_value_delimiter=true)]
    pub topics: Vec<String>,

    /// number of images to download per-class
    #[arg(short, long, default_value_t = 20)]
    pub nsamples: usize,

    /// name of directory to save to
    #[arg(short, long, default_value = "images")]
    pub dir: String,
}

struct TopicHandler<'a> {
    topic: &'a str,
    download_manager: DLManager,
}

impl<'a> TopicHandler<'a> {
    pub async fn download_topic(&mut self, dir: String) -> u8 {
        let nsamples = self.download_manager.target_size;

        //download path = base_dir + topic
        let dir = format!("{}/{}", dir, self.topic);

        let batch = Fetcher::query_api(self.topic, nsamples).await;
        let total = self.download_manager.download_batch(batch, &dir).await;
        total
    }
}

pub struct TinyDataClient {
    args: Args,
    multi_progress_bar: MultiProgress,
}

impl TinyDataClient {
    pub fn new(args: Args) -> Self {
        Self {
            args,
            multi_progress_bar: MultiProgress::new(),
        }
    }

    pub async fn run(&mut self) {
        let args = self.args.clone();

        //create image directories
        for topic in &args.topics {
            let dir = format!("{}/{}", args.dir, topic);
            create_dir_all(&dir).expect("failed to create directory");
        }

        let nsamples = args.nsamples;
        // need thread-safe string for async
        let dir = Arc::new(args.dir);

        let futures: Vec<JoinHandle<u8>> = args
            .topics
            .into_iter()
            .map(|topic| {
                let dir = Arc::clone(&dir);
                let mut pb = self
                    .multi_progress_bar
                    .add(ProgressBar::new(nsamples as u64));

                //style
                stylize_pb(&mut pb, &topic);

                tokio::spawn(async move {
                    let download_manager = DLManager::new(nsamples, pb);

                    let mut topic_handler = TopicHandler {
                        topic: &topic,
                        download_manager,
                    };
                    topic_handler.download_topic(dir.to_string()).await
                })
            })
            .collect();

        //time execution
        let now = Instant::now();
        let total = join_all(futures).await;
        let elapsed = now.elapsed();

        let total: u8 = total.into_iter().map(|res| res.unwrap()).sum();

        println!(
            "{}/{} files saved successfully to `./{}` in {}s 📦",
            total,
            self.args.nsamples * self.args.topics.len(),
            self.args.dir,
            elapsed.as_secs(),
        );
    }
}

fn stylize_pb(pb: &mut ProgressBar, name: &str) {
    let default = "{msg} {spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})";
    pb.set_style(
        ProgressStyle::with_template(default)
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(String::from(name));
}
