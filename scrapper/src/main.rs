use std::{fs::read_to_string, vec}; // use instead of std::fs::File
use std::path::Path;
use std::fs::File;

use futures::stream::StreamExt;
use futures::TryStreamExt;

use serde::{Serialize, Deserialize};
use serde_json::{Map, json};

use reqwest::Client;

use select::document::Document;
use select::predicate::{Attr, Class, Name, Predicate};
use select::node::Node;

#[derive(Debug, Serialize, Deserialize)]
struct Movie {
    title: String,
    data: Vec<String>,
    kinds: Vec<String>,
    directors: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct User {
    rating: u32,
    review: u32,
    subscriber: u32,
}

async fn extract_user_data(html: &str) -> User {
    let div: String = Document::from_read(html.as_bytes())
        .unwrap()
        .find(Class("stats-overview").descendant(Name("div")))//.descendant(Class("stats-overview-item-value")))
        //.filter(|n| n.attr("class").map_or(false, |v| v == ("stats-overview-item-value")))
        .map(|n| n.text())
        .collect();

    if div.len() == 0 {
        return User {
            rating: 0,
            review: 0,
            subscriber: 0,
        }
    }

    //println!("len : {}", div.len());


    let mut ws = div.split_whitespace();
    for i in 0..4 {
        ws.next();
    }
    let mut data: Vec<u32> = vec![];
    for _ in 0..3 {
        let t = ws.next()
            .unwrap()
            .parse::<u32>()
            .unwrap();
        data.push(t);
        ws.next();
        //println!("{:?} : {}", ws.next(), t);
    }

    
    User {
        rating: data[0],
        review: data[1],
        subscriber: data[2],
    }
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

async fn extract_kinds(html: &str) -> Vec<String> {
    let kinds: Vec<String> = Document::from_read(html.as_bytes())
        .unwrap()
        .find(Class("meta-body-info").descendant(Name("span")))
        .filter(|n| n.attr("class").map_or(false, |v| v.contains("ACrL2ZACrpbG1")))
        .map(|n| n.text())
        .collect();

    //println!("kinds: {:?}", kinds);

    kinds
}

async fn extract_directors(html: &str) -> Vec<String> {
    let directors: Vec<String> = Document::from_read(html.as_bytes())
        .unwrap()
        .find(Class("meta-body-direction").descendant(Name("span")))
        .filter(|n| n.attr("class").map_or(false, |v| v.contains("ACrL3")))
        .map(|n| n.text())
        .collect();

    //println!("director: {:?}", directors);

    directors
}

async fn movie_pipeline(html: &str) -> Movie {
    let title = extract_title(&html);
    let data = extract_avg_note(&html);
    let kinds = extract_kinds(&html);
    let directors = extract_directors(&html);
    let (title, data, kinds, directors) = futures::join!(title, data, kinds, directors);


    Movie {
        title,
        data,
        kinds,
        directors,
    }
}

async fn user_pipeline(html: &str) -> User {
    extract_user_data(html).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    let users_path = Path::new("../data/json/users.json");
    let users_str = read_to_string(users_path).expect("file not found");
    let users : Vec<String> = serde_json::from_str(&users_str).expect("error while reading json");
    let nn = users.len();
    let n = 100_000;

    let client = Client::builder().build()?;
    let resp = futures::stream::iter(users[n..nn].to_vec())
    .map(|user| {
        let url = format!("https://www.allocine.fr/membre-{}", user);

        let send_fut = client.get(&url).send();


        async move {
            let text = send_fut.await?.text().await?;
            //println!("id {}", user);
            let data = user_pipeline(&text).await;

            reqwest::Result::Ok((user, data))
        }
    })
    .buffer_unordered(50)
    .try_collect::<Vec<(String, User)>>()
    .await;

    let mut map = Map::new();

    match resp {
        Ok(data) => {
            for chunk in data {
                map.insert(chunk.0, json!(chunk.1));
            }
        }
        Err(e) => println!("failed, {}", e),
    };

    serde_json::to_writer(&File::create("users_additional_3.json")?, &map)?;

    Ok(())
}



// let movies_path = Path::new("../data/json/movies.json");
// let movies_str = read_to_string(movies_path).expect("file not found");
// let movies : Vec<String> = serde_json::from_str(&movies_str).expect("error while reading json");

// let client = Client::builder().build()?;
// let resp = futures::stream::iter(movies)
// .map(|movie| {
//     let url = format!("https://www.allocine.fr/film/fichefilm_gen_cfilm={}.html", movie);
    
//     let send_fut = client.get(&url).send();


//     async move {
//         let text = send_fut.await?.text().await?;
//         let data = pipeline(&text).await;
//         //println!("movie_id: {}", movie);
        
//         reqwest::Result::Ok((movie, data))
//     }
// })
// .buffer_unordered(50)
// .try_collect::<Vec<(String, Movie)>>()
// .await;

// let mut map = Map::new();

// match resp {
//     Ok(data) => {
//         for chunk in data {
//             map.insert(chunk.0, json!(chunk.1));
//         }
//     }
//     Err(_) => println!("failed"),
// };

// serde_json::to_writer(&File::create("movie_additional_full.json")?, &map)?;

// Ok(())