use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::*;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = arr1(&[4., 5., 3., 1.]);
    let w = arr1(&[4., 5., 3., 1.]);
    let output = meaures_of_similarity(v, w);
    assert_eq!(output.pearsons_corr, 1.);
    dbg!(start_time.elapsed());
}
