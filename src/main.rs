use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::*;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = arr1(&[4., 5., 3.]);
    let w = arr1(&[5., 4., 6.]);
    let vcts = vec![v, w];
    let weights = vec![10.0, 69.0];
    let weight_mul = mul_weights(vcts, weights);
    dbg!(weight_mul);
    dbg!(start_time.elapsed());
}
