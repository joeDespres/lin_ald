use matrix::MyMatrixMethods;
use ndarray::prelude::*;
use std::time::Instant;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();
    let d = 50;
    let p = 51;
    let a = Array2::from_shape_vec((d, p), vec::gen_brownian_motion(d * p).to_vec()).unwrap();
    let b = Array2::from_shape_vec((p, d), vec::gen_brownian_motion(p * d).to_vec()).unwrap();
    let c = matrix::mat_mul(a.clone(), b.clone());
    let e = a.dot(&b);
    dbg!(c.shape());
    dbg!(e.shape());
    let diff = c - e;
    assert!(diff.max() < vec::TOL);
    dbg!(start_time.elapsed());
}
