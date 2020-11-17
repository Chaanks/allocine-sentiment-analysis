use std::fs::read_to_string; // use instead of std::fs::File
use std::path::Path;
use std::fs::File;

use futures::stream::StreamExt;
use futures::TryStreamExt;

use serde::{Serialize, Deserialize};
use serde_json::{Map, json};

use reqwest::Client;

use select::document::Document;
use select::predicate::Name;

#[derive(Debug, Serialize, Deserialize)]
struct Movie {
    title: String,
    data: Vec<String>
}

async fn extract_title(html: &str) -> String {
    let title: String = Document::from_read(html.as_bytes())
        .unwrap()
        .find(Name("meta"))
        .filter(|n| n.attr("property").map_or(false, |v| v == "og:title"))
        .filter_map(|n| n.attr("content"))
        .collect();
    
    title
}

async fn extract_avg_note(html: &str) -> Vec<String> {
    let mut div = Document::from_read(html.as_bytes())
        .unwrap()
        .find(Name("div"))
        .filter(|n| n.attr("class").map_or(false, |v| v.contains("rating-holder")))
        .map(|n| n.inner_html())
        .collect::<Vec<String>>();
    
    if div.is_empty() {
        return vec!["Null".to_string()];
    }

    let score = Document::from_read(div.remove(0).as_bytes())
        .unwrap()
        .find(Name("span"))
        .filter(|n| n.attr("class").is_some())
        .map(|n| n.text())
        .collect::<Vec<String>>();

    score
}


async fn pipeline(html: &str) -> Movie {
    let title = extract_title(&html);
    let data = extract_avg_note(&html);
    let (title, data) = futures::join!(title, data);

    Movie {
        title,
        data,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let movies_path = Path::new("../data/json/movies.json");
    let movies_str = read_to_string(movies_path).expect("file not found");
    let movies : Vec<String> = serde_json::from_str(&movies_str).expect("error while reading json");

    let client = Client::builder().build()?;
    let resp = futures::stream::iter(movies)
    .map(|movie| {
        let url = format!("https://www.allocine.fr/film/fichefilm_gen_cfilm={}.html", movie);
        let send_fut = client.get(&url).send();
        async move {
            let text = send_fut.await?.text().await?;
            let data = pipeline(&text).await;
            
            reqwest::Result::Ok((movie, data))
        }
    })
    .buffer_unordered(50)
    .try_collect::<Vec<(String, Movie)>>()
    .await;

    let mut map = Map::new();

    match resp {
        Ok(data) => {
            for chunk in data {
                map.insert(chunk.0, json!(chunk.1));
            }
        }
        Err(_) => println!("failed"),
    };

    serde_json::to_writer(&File::create("movie_additional.json")?, &map)?;

    Ok(())
}