use matrix::MyMatrixMethods;
use ndarray::{prelude::*, ShapeError};
use std::time::Instant;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();
    let d = 5;

    let a = Array::from_shape_vec((d, d), vec::gen_brownian_motion(d * d).to_vec()).unwrap();
    let symetric = jankey_avg(a);
    dbg!(symetric);

    dbg!(start_time.elapsed());
}
fn jankey_avg(a: Array2<f64>) -> Array2<f64> {
    let a_t = &a.t();
    a_t + &a / 2.
}
fn test_jankey_avg() {
    let d = 5;
    let a = Array::from_shape_vec((d, d), vec::gen_brownian_motion(d * d).to_vec()).unwrap();
    let symetric = jankey_avg(a);

    assert!(matrix::is_symmetric(symetric));
}
