use ndarray::prelude::*;
use std::{process::Output, time::Instant};

mod metd;
use crate::metd::FloatMethods;
mod vec;

fn main() {
    let start_time = Instant::now();
    let v = vec::gen_brownian_motion(10);
    let gauss = arr1(&[0., 0.1, 0.3, 0.8, 0.3, 0.1]);
    dbg!(gauss.l2_norm());
    dbg!(gauss.set_magnitude(0.5));
    let kernel = vec::unit_vec(gauss);
    dbg!(kernel);

    dbg!(start_time.elapsed());
}
