use std::collections::HashMap;

use nalgebra::SMatrix;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rand_distr::{Distribution, Normal};

use crate::{
    common,
    constant::{GRID_SIZE, NUM_OF_PROJECTIONS},
};

pub fn generate_projection_matrix() -> Vec<SMatrix<f32, GRID_SIZE, 1>> {
    let mut rng = ChaCha20Rng::from_rng(&mut rand::rng());

    let std_dev = 1.0 / (GRID_SIZE as f64).sqrt();
    let normal = Normal::new(0.0, std_dev).unwrap();

    (0..NUM_OF_PROJECTIONS)
        .map(|_| {
            let mut projection = SMatrix::<f32, GRID_SIZE, 1>::zeros();
            for i in 0..GRID_SIZE {
                projection[(i, 0)] = normal.sample(&mut rng) as f32;
            }
            projection
        })
        .collect()
}

pub fn generate_fingerprint_database(
    projections: Vec<SMatrix<f32, GRID_SIZE, 1>>,
) -> HashMap<usize, (usize, Vec<f32>)> {
    let mut fingerprint_database = HashMap::new();

    for (i, projection) in projections.iter().enumerate() {
        let mut fingerprint = Vec::with_capacity(GRID_SIZE);
        for j in 0..GRID_SIZE {
            let observation = projection[(j, 0)];
            fingerprint.push(observation);
        }
        fingerprint_database.insert(i, (i, fingerprint));
    }

    fingerprint_database
}

pub fn predict_location_from_database(
    fingerprint_database: &HashMap<usize, (usize, Vec<f32>)>,
    observation: (usize, f32),
) -> usize {
    let ideal_observation = fingerprint_database.get(&observation.0).unwrap();
    let mut min_distance = f32::MAX;
    let mut predicted_location = None;
    for (i, ideal) in ideal_observation.1.iter().enumerate() {
        let distance = (ideal - observation.1).abs();
        if distance < min_distance {
            min_distance = distance;
            predicted_location = Some(i);
        }
    }

    predicted_location.unwrap()
}

pub fn predict_location_from_database_quantized(
    fingerprint_database: &HashMap<usize, (usize, Vec<f32>)>,
    observation: (usize, i8),
) -> Vec<usize> {
    let ideal_observation = fingerprint_database.get(&observation.0).unwrap();
    let locations: Vec<usize> = ideal_observation
        .1
        .iter()
        .map(|o| common::quantize(*o))
        .enumerate()
        .filter(|(_, quantized)| *quantized == observation.1)
        .map(|(i, _)| i)
        .collect();

    locations
}
