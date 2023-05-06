use matrix::MyMatrixMethods;
use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::gen_brownian_motion;
mod matrix;
mod vec;

fn main() {
    for i in 50..900000 {
        if i % 150 == 0 {
            dbg!(i);
            let d = i;
            let a = Array::from_shape_vec((d, d), gen_brownian_motion(d * d).to_vec()).unwrap();
            let b = Array::from_shape_vec((d, d), gen_brownian_motion(d * d).to_vec()).unwrap();
            let start_time = Instant::now();
            let _ = matrix::mat_mul(a.clone(), b.clone());
            dbg!(start_time.elapsed());
            let start_time = Instant::now();
            let _ = a.dot(&b);
            dbg!(start_time.elapsed());
        }
    }
}
