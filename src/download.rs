use futures::future::join_all;
use indicatif::ProgressBar;
use reqwest::{IntoUrl, StatusCode};
// use std::fs::File;
// use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use tokio::fs::File as AsyncFile;
use tokio::io::AsyncWriteExt;

pub struct Task {
    pub downloader: Arc<Downloader>,
    pub url: Option<String>, //so i can move it later
}

impl Task {
    pub async fn download(&mut self, filename: String) -> Result<u8, Box<dyn std::error::Error>> {
        let url = self.url.take().unwrap();
        let downloader = self.downloader.clone();
        let res = downloader.download(url, filename).await?;
        Ok(res)
    }
}

//shared state for each download task
pub struct Downloader {
    pub cur: Mutex<usize>,
    pub progress_bar: Mutex<ProgressBar>,
}

impl Downloader {
    pub fn new(progress_bar: Mutex<ProgressBar>) -> Self {
        Self {
            cur: Mutex::new(0),
            progress_bar,
        }
    }

    pub async fn download(
        &self,
        url: impl IntoUrl,
        filename: String,
    ) -> Result<u8, Box<dyn std::error::Error>> {
        let res = reqwest::get(url).await?;
        match res.status() {
            StatusCode::OK => {
                let bytes = res.bytes().await?;

                // ad-hoc
                if !bytes.starts_with(b"<!DOCTYPE html>") {
                    let mut file = AsyncFile::create(filename).await?;
                    file.write_all(&bytes).await?;
                    // let mut file = File::create(filename)?;
                    // file.write_all(&bytes)?;

                    //interior mutability + async-safe lock access !
                    {
                        *self.cur.lock().unwrap() += 1;
                        self.progress_bar.lock().unwrap().inc(1);
                    }
                }
            }
            _ => return Ok(0),
        }

        Ok(1)
    }
}

pub struct DLManager {
    pub target_size: usize,
    pub downloader: Arc<Downloader>,
}

impl DLManager {
    pub fn new(target_size: usize, progress_bar: ProgressBar) -> Self {
        let downloader = Arc::new(Downloader::new(Mutex::new(progress_bar)));

        DLManager {
            target_size,
            downloader,
        }
    }

    //TODO: add upper limit on batch?
    pub async fn download_batch<'a>(&mut self, batch: Vec<String>, dir: &'a str) -> u8 {
        let cur = *self.downloader.cur.lock().unwrap();

        // if cur == self.target_size { // used in old streaming-like approach
        //     return true;
        // }

        // so we don't overflow on the quota
        let how_many = usize::min(self.target_size - cur, batch.len());
        // println!("taking {}", how_many);

        let mut futures = vec![];
        for (id, url) in batch.into_iter().take(how_many).enumerate() {
            let id = id + cur;

            let mut task = Task {
                downloader: self.downloader.clone(),
                url: Some(url.to_string()),
            };

            //now spawn the batch and await
            let filename = format!("{}/{}.jpeg", dir, id);

            futures.push(tokio::spawn(async move {
                match task.download(filename).await {
                    Ok(i) => i,
                    _ => 0,
                }
            }));
        }
        let did_download = join_all(futures).await;
        let total: u8 = did_download.into_iter().map(|res| res.unwrap()).sum();
        total
    }
}
