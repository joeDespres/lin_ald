use std::time::Instant;

use ndarray::Array2;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    let a = matrix::bm_mat(10);
    let b = matrix::bm_mat(10);

    let norm = matrix::frobenius_norm_to_zero(a, b);

    dbg!(start_time.elapsed());
    dbg!(norm);
}
