use crate::lin_ops::MyLinOps;
use ndarray::arr2;
use std::time::Instant;
mod lin_ops;
mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    dbg!(a.cofactors_mat());
    // let det_a = a.det().unwrap();
    dbg!(start_time.elapsed());
}
