use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rand::seq::IndexedRandom;
use rayon::prelude::*;

use crate::constant::NUM_OF_PROJECTIONS;

pub mod client;
pub mod common;
pub mod constant;
pub mod server;

fn main() {
    // Set larger stack size for threads
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024) // 8MB stack
        .build_global()
        .unwrap();
    
    println!("Generating projections...");
    let projections = Arc::new(server::generate_projection_matrix());

    println!("Generating fingerprint database...");
    let fingerprint_database = server::generate_fingerprint_database((*projections).clone());

    let grid_candidates: Vec<(usize, usize)> = vec![(5, 5), (120, 50), (200, 200), (10, 5)];

    let count = Mutex::new(HashMap::<usize, usize>::new());
    
    (0..NUM_OF_PROJECTIONS * 3)
        .into_par_iter()
        .for_each(|i| {
            let mut thread_rng = rand::rng();
            let current_grid = *grid_candidates.choose(&mut thread_rng).unwrap();
            
            let thread_projections = Arc::clone(&projections);
            let mut thread_client = client::Client::new_with_ref(i % NUM_OF_PROJECTIONS, &thread_projections);
            let (index, observation) = thread_client.observe_quantized(current_grid);
            println!("Observation {i}: index = {index}, observation = {observation}");
            
            let predictions = server::predict_location_from_database_quantized(
                &fingerprint_database,
                (index, observation),
            );
            
            let mut count_guard = count.lock().unwrap();
            for location in predictions.iter() {
                count_guard.entry(*location).and_modify(|e| *e += 1).or_insert(1);
            }
        });
    
    let count = count.into_inner().unwrap();

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
