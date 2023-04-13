use ndarray::prelude::*;
use std::time::Instant;

mod vec;

fn main() {
    let start_time = Instant::now();
    let _unit_vect = arr1(&[2., 2., 3.]);
    let x = arr1(&[3.0, 4.0, 3.0]);
    let _result = vec::unit_vec(x);
    dbg!(start_time.elapsed());
}
