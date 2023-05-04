use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::gen_brownian_motion;
mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();
    let d = 5;
    let a = Array::from_shape_vec((d, d), gen_brownian_motion(d * d).to_vec()).unwrap();
    let b = Array::from_shape_vec((d, d), gen_brownian_motion(d * d).to_vec()).unwrap();
    dbg!(start_time.elapsed());
    dbg!(a, b);
}
