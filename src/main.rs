use ndarray_linalg::Determinant;
use std::time::Instant;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();
    let rows = 100;
    let cols = 100;

    let a = matrix::bm_mat_rec(rows, cols);
    let det_a = a.det().unwrap();
    dbg!(det_a);

    let rows = 10000;
    let cols = 69;

    let b = matrix::bm_mat_rec(rows, cols);
    let c = a.dot(&b);
    dbg!(c.dim());

    dbg!(start_time.elapsed());
}
