use nalgebra::DVector;

use crate::{
    common,
    constant::{GRID_SIZE, NUM_OF_PROJECTIONS},
};

pub struct Client {
    pub i: usize,
    pub projection_matrix: Vec<DVector<f32>>,
}

pub struct ClientRef<'a> {
    pub i: usize,
    pub projection_matrix: &'a Vec<DVector<f32>>,
}

impl Client {
    pub fn new(i: usize, projection_matrix: Vec<DVector<f32>>) -> Self {
        Client {
            i,
            projection_matrix,
        }
    }
    
    pub fn new_with_ref(i: usize, projection_matrix: &Vec<DVector<f32>>) -> ClientRef {
        ClientRef {
            i,
            projection_matrix,
        }
    }

    pub fn observe(&mut self, current_grid: (usize, usize)) -> (usize, f32) {
        let index = common::grid_to_index(current_grid);
        let projection = &self.projection_matrix[self.i];

        let observation = projection[index];

        let result = (self.i, observation);

        if self.i < NUM_OF_PROJECTIONS - 1 {
            self.i += 1;
        } else {
            self.i = 0;
        }

        result
    }

    pub fn observe_quantized(&mut self, current_grid: (usize, usize)) -> (usize, i8) {
        let normal_observation = self.observe(current_grid);
        let quantized_observation = common::quantize(normal_observation.1);

        (normal_observation.0, quantized_observation)
    }
}

impl<'a> ClientRef<'a> {
    pub fn observe(&mut self, current_grid: (usize, usize)) -> (usize, f32) {
        let index = common::grid_to_index(current_grid);
        let projection = &self.projection_matrix[self.i];

        let observation = projection[index];

        let result = (self.i, observation);

        if self.i < NUM_OF_PROJECTIONS - 1 {
            self.i += 1;
        } else {
            self.i = 0;
        }

        result
    }

    pub fn observe_quantized(&mut self, current_grid: (usize, usize)) -> (usize, i8) {
        let normal_observation = self.observe(current_grid);
        let quantized_observation = common::quantize(normal_observation.1);

        (normal_observation.0, quantized_observation)
    }
}
