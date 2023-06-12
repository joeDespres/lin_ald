use crate::lin_ops::MyLinOps;
use ndarray::arr2;
use std::time::Instant;
mod lin_ops;
mod matrix;
mod vec;
use ndarray_linalg::Solve;

fn main() {
    let start_time = Instant::now();

    let a = arr2(&[[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]]);
    dbg!(a.invert());
    let target = arr2(&[[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]);
    // let det_a = a.det().unwrap();
    dbg!(start_time.elapsed());
}
