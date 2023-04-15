use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::orth_decomp;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = arr1(&[4., 5., 3.]);
    let w = arr1(&[5., 4., 6.]);
    let beta = orth_decomp(&v, w);
    dbg!(beta);

    dbg!(start_time.elapsed());
}
