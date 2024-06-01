use futures::future::join_all;
use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use std::fs::create_dir_all;
use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinHandle;

use crate::download::*;
use crate::fetch::*;

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
    multi_progress_bar: MultiProgress,
}

impl TinyDataClient {
    pub fn new() -> Self {
        Self {
            multi_progress_bar: MultiProgress::new(),
        }
    }

    pub async fn run(&self, topics: Vec<String>, nsamples: usize, dir: String) {
        //create image directories
        for topic in &topics {
            let dir = format!("{}/{}", dir, topic);
            create_dir_all(&dir).expect("failed to create directory");
        }

        let dir = Arc::new(dir);
        let topics_len = topics.len();

        let futures: Vec<JoinHandle<u8>> = topics
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
            "📦 {}/{} files saved successfully to `./{}` in {}s",
            total,
            nsamples * topics_len,
            dir,
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
