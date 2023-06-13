use crate::lin_ops::MyLinOps;
use ndarray::arr2;
use std::time::Instant;
mod lin_ops;
mod matrix;
mod vec;

fn main() {
    for i in 25..100000 {
        dbg!(i);
        let start_time = Instant::now();
        let a = matrix::bm_mat(i);
        let _ = a.invert();
        dbg!(start_time.elapsed());
    }
}
