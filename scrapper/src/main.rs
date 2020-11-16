use std::fs::read_to_string; // use instead of std::fs::File
use std::path::Path;
use std::collections::HashMap;
use std::io::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::{thread, time};
use std::future::Future;
use std::task::Poll;
use std::time::{Duration, Instant};

use tokio::task;

use futures::stream::StreamExt;
use futures::executor::block_on;
use futures::TryStreamExt;

use serde::{Serialize, Deserialize};
use serde_json::{Map, Value};

use reqwest::Client;

use select::document::Document;
use select::predicate::Name;


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
    let note: Vec<String> = Document::from_read(html.as_bytes())
        .unwrap()
        .find(Name("span"))
        .filter(|n| n.attr("class").map_or(false, |v| v == "stareval-note"))
        .map(|n| n.text())
        .collect::<Vec<String>>();
        //.remove(1);

    // let sleep_time = time::Duration::from_millis(2000);
    // thread::sleep(sleep_time);
    // println!("Movie average note : {:?}", note);
    note
}


async fn pipeline(html: &str) -> String {
    let title = extract_title(&html);
    //let score = extract_avg_note(&html);
    //futures::join!(title, score);

    title.await
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
            //println!("run pipeline");
            let data = pipeline(&text).await;
            
            reqwest::Result::Ok((movie, data))
        }
    })
    .buffer_unordered(50)
    .try_collect::<Vec<(String, String)>>()
    .await;

    let mut map = Map::new();

    match resp {
        Ok(data) => {
            for chunk in data {
                map.insert(chunk.0, Value::String(chunk.1));
            }
        }
        Err(_) => println!("failed"),
    };

    
    //println!("{}", serde_json::to_string_pretty(&map).unwrap());

    serde_json::to_writer(&File::create("movie_mapping.json")?, &map)?;

    Ok(())
}