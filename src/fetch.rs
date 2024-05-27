use futures::future::join_all;
use regex::Regex;
use std::env;

const GOOGLE_RESULT_SIZE: usize = 10;
const GOOGLE_ENDPOINT: &'static str = "https://www.googleapis.com/customsearch/v1";

fn get_creds() -> (String, String) {
    let api_key =
        env::var("CUSTOM_SEARCH_API_KEY").expect("variable `CUSTOM_SEARCH_API_KEY` not set");
    let search_engine_id =
        env::var("SEARCH_ENGINE_ID").expect("variable `SEARCH_ENGINE_ID` not set");

    (api_key, search_engine_id)
}

enum SearchType {
    Image,
}

#[allow(dead_code)]
pub struct Params {
    cx: String,
    key: String,
    search_type: SearchType,
    q: String,
    num: usize,
    start: usize,
}

impl Params {
    pub fn new() -> Self {
        let (key, cx) = get_creds();
        Params {
            cx,
            key,
            search_type: SearchType::Image,
            q: String::new(),
            num: 10, //max num
            start: 0,
        }
    }

    pub fn to_list(self) -> Vec<(String, String)> {
        vec![
            ("cx".to_string(), self.cx),
            ("key".to_string(), self.key),
            ("searchType".to_string(), "image".to_string()), //bleh
            ("q".to_string(), self.q),
            ("num".to_string(), self.num.to_string()),
            ("start".to_string(), self.start.to_string()),
        ]
    }
}

pub struct Fetcher {
    // tx: Arc<mpsc::Sender<Vec<String>>>,
}

impl Fetcher {
    pub async fn query_api(query: &str, n: usize) -> Vec<String> {
        //guesstimate how many api calls we need to make
        let maybe_one = if n % GOOGLE_RESULT_SIZE > 0 { 1 } else { 0 };
        let nqueries = n / GOOGLE_RESULT_SIZE + maybe_one + 1; //+1 guesstimate
        let nqueries = usize::min(nqueries, 10);

        let mut futures = vec![];
        for i in 0..nqueries {
            //from specs
            let mut offset = 0;
            if i > 0 {
                offset = 10 * i + 1;
            }

            //build params
            let mut params = Params::new();
            params.start = offset;
            params.q = format_query(query);

            futures.push(tokio::spawn(async move {
                Self::google_search(params).await.unwrap()
            }));
        }

        let urls: Vec<Vec<String>> = join_all(futures)
            .await
            .into_iter()
            .map(|x| x.unwrap())
            .collect();

        let urls = urls.into_iter().flatten().collect();
        urls
    }

    // could make this a streaming function
    async fn google_search(params: Params) -> Result<Vec<String>, reqwest::Error> {
        let params = params.to_list();

        let client = reqwest::Client::new();
        let res = client.get(GOOGLE_ENDPOINT).query(&params).send().await?;
        let body = res.text().await?;

        let urls = extract_urls(&body);
        Ok(urls)
    }
}

fn format_query(topic: &str) -> String {
    //remove underscores
    let new_topic: Vec<String> = topic.split('_').map(String::from).collect();
    new_topic.join(" ")
}

// from chat
fn extract_urls(text: &str) -> Vec<String> {
    // Define the regex pattern to match the URLs
    let re = Regex::new(r#""link":\s*"(https?://[^"]*)""#).unwrap();

    // Initialize a vector to store the extracted URLs
    let mut urls = Vec::new();

    // Iterate over all matches in the text
    for cap in re.captures_iter(text) {
        // Capture group 1 contains the URL
        if let Some(url) = cap.get(1) {
            urls.push(url.as_str().to_string());
        }
    }

    urls
}

//dummy
// fn get_urls() -> Vec<String> {
//     let urls = fs::read_to_string("urls.txt").expect("couldn't read urls");
//     let urls: Vec<String> = urls.lines().map(|s| s.to_string()).collect();
//     urls
// }
// pub async fn stream_batched(&self) {
//     let urls = get_urls();
//     let urls = urls.as_slice();
//
//     sleep(Duration::from_millis(500)).await;
//
//     let mut buffer = Vec::new();
//     for batch in urls.windows(10).step_by(10) {
//         buffer.extend_from_slice(batch);
//
//         if buffer.len() >= 150 {
//             // println!("sent: {}", buffer.len());
//             match self.tx.send(buffer.clone()).await {
//                 Ok(_) => (),
//                 _ => return (),
//             }
//             buffer.clear();
//         }
//     }
//     if !buffer.is_empty() {
//         // println!("sent: {}", buffer.len());
//         match self.tx.send(buffer.clone()).await {
//             Ok(_) => (),
//             _ => return (),
//         }
//         buffer.clear();
//     }
// }
