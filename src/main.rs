use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::*;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = Array::range(-50., 50., 1.);
    let w = Array::range(-50., 50., 1.);
    let output = meaures_of_similarity(v, w);
    assert_eq!(output.pearsons_corr, 1.);
    dbg!(start_time.elapsed());
}
