use nalgebra::SMatrix;

use crate::{
    common,
    constant::{GRID_SIZE, NUM_OF_PROJECTIONS},
};

pub struct Client {
    pub i: usize,
    pub projection_matrix: Vec<SMatrix<f32, GRID_SIZE, 1>>,
}

impl Client {
    pub fn new(i: usize, projection_matrix: Vec<SMatrix<f32, GRID_SIZE, 1>>) -> Self {
        Client {
            i,
            projection_matrix,
        }
    }

    pub fn observe(&mut self, current_grid: (usize, usize)) -> (usize, f32) {
        let index = common::grid_to_index(current_grid);
        let projection = &self.projection_matrix[self.i];

        let observation = projection[(index, 0)];

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
