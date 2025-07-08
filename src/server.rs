use std::collections::HashMap;

use nalgebra::DVector;
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::{
    common,
    constant::{GRID_SIZE, NUM_OF_PROJECTIONS},
};

pub fn generate_projection_matrix() -> Vec<DVector<f32>> {
    let std_dev = 1.0 / (GRID_SIZE as f64).sqrt();
    let normal = Normal::new(0.0, std_dev).unwrap();

    (0..NUM_OF_PROJECTIONS)
        .into_par_iter()
        .map(|i| {
            let mut rng = ChaCha20Rng::from_seed([i as u8; 32]);
            let mut projection = DVector::<f32>::zeros(GRID_SIZE);
            for j in 0..GRID_SIZE {
                projection[j] = normal.sample(&mut rng) as f32;
            }
            projection
        })
        .collect()
}

pub fn generate_fingerprint_database(
    projections: Vec<DVector<f32>>,
) -> HashMap<usize, (usize, Vec<f32>)> {
    projections
        .into_par_iter()
        .enumerate()
        .map(|(i, projection)| {
            let mut fingerprint = Vec::with_capacity(GRID_SIZE);
            for j in 0..GRID_SIZE {
                let observation = projection[j];
                fingerprint.push(observation);
            }
            (i, (i, fingerprint))
        })
        .collect()
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
