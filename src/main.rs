use ndarray::prelude::*;
use std::time::Instant;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v: Array2<f64> = Array::zeros((4, 1));
    let output = vec::reorient_vec(v);
    dbg!(output);

    dbg!(start_time.elapsed());
}
