use std::collections::HashMap;

use rand::seq::IndexedRandom;

use crate::constant::NUM_OF_PROJECTIONS;

pub mod client;
pub mod common;
pub mod constant;
pub mod server;

fn main() {
    println!("Generating projections...");
    let projections = server::generate_projection_matrix();

    println!("Generating fingerprint database...");
    let fingerprint_database = server::generate_fingerprint_database(projections.clone());

    let mut client = client::Client::new(0, projections);

    let grid_candidates: Vec<(usize, usize)> = vec![(5, 5), (120, 50), (200, 200), (10, 5)];

    let mut rng = rand::rng();

    let mut count: HashMap<usize, usize> = HashMap::new();
    for i in 0..NUM_OF_PROJECTIONS * 3 {
        let current_grid = *grid_candidates.choose(&mut rng).unwrap();

        let (index, observation) = client.observe_quantized(current_grid);
        println!("Observation {i}: index = {index}, observation = {observation}");
        let predictions = server::predict_location_from_database_quantized(
            &fingerprint_database,
            (index, observation),
        );
        for location in predictions.iter() {
            count.entry(*location).and_modify(|e| *e += 1).or_insert(1);
        }
    }

    let sorted = count.iter().collect::<Vec<_>>();
    let mut sorted = sorted
        .into_iter()
        .map(|(k, v)| (*k, *v))
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Top 10 locations:");
    for (i, (location, count)) in sorted.iter().take(10).enumerate() {
        println!(
            "Rank {}: Location: {:?}, Count: {count}",
            i + 1,
            common::index_to_grid(*location)
        );
    }
}
