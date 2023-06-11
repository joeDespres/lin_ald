use crate::lin_ops::MyLinOps;
use ndarray::arr2;
use std::time::Instant;
mod lin_ops;
mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    let a = arr2(&[[2.0, 1.0, 2.0], [-2.0, 2.0, 1.0], [1.0, 2.0, -2.0]]);

    dbg!(a.minors_mat());

    // let det_a = a.det().unwrap();
    dbg!(start_time.elapsed());
}
