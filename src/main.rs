use std::time::Instant;

use ndarray::Array2;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();
    let rows = 10;
    let cols = 10000;

    let a = matrix::bm_mat_rec(rows, cols);

    let rows = 10000;
    let cols = 69;

    let b = matrix::bm_mat_rec(rows, cols);
    let c = a.dot(&b);
    dbg!(c.dim());

    dbg!(start_time.elapsed());
}
