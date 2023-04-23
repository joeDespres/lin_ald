use ndarray::{concatenate, prelude::*};
use std::time::Instant;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = series(30);
    dbg!(v);
    dbg!(start_time.elapsed());
}

fn series(n: usize) -> Array1<i32> {
    let v: Array1<i32> = -1 * Array::ones(n / 3);
    let w: Array1<i32> = Array::ones(n / 3);
    let y: Array1<i32> = -1 * Array::ones(n / 3);
    concatenate(Axis(0), &[v.view(), w.view(), y.view()]).unwrap()
}
