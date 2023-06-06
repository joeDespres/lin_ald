use ndarray::Axis;
use std::time::Instant;

use crate::matrix::{cov_mat, cov_to_corr};
use ndarray::arr2;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();
    let rows = 1000;
    let cols = 10;

    let a = arr2(&[[2., 2.], [3., 1.], [4., 3.]]);
    let c = cov_mat(a.clone());
    let r = cov_to_corr(c.clone());
    dbg!(start_time.elapsed());
}
