use crate::vec::*;
use ndarray::prelude::*;

use std::time::Instant;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = normal_vec(100000);
    let w = normal_vec(100000);

    for _ in 1..1000 {
        let _dot = v.dot(&w);
    }
    dbg!(start_time.elapsed());
}
