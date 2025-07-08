use crate::constant::{GRID_SIZE, GRID_WIDTH};

pub fn quantize(value: f32) -> i8 {
    let std_dev = 1.0 / (GRID_SIZE as f32).sqrt();

    let three_sigma = 3.0 * std_dev;

    let scale_factor = 120.0 / three_sigma;

    let scaled = value * scale_factor;
    let clamped = scaled.clamp(-128.0, 127.0);

    clamped.round() as i8
}

pub fn grid_to_index(grid: (usize, usize)) -> usize {
    grid.0 * GRID_WIDTH + grid.1
}

pub fn index_to_grid(index: usize) -> (usize, usize) {
    let x = index / GRID_WIDTH;
    let y = index % GRID_WIDTH;
    (x, y)
}
