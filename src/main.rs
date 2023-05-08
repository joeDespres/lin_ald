use matrix::MyMatrixMethods;
use ndarray::{prelude::*, ShapeError};
use std::time::Instant;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    dbg!(start_time.elapsed());
}
