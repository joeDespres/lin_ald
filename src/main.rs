use crate::vec::*;
use ndarray::prelude::*;
use ndarray::stack;
use ndarray_stats::CorrelationExt;

use std::time::Instant;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = normal_vec(100000);
    let w = normal_vec(100000);

    println!("my trash code");
    let start_time = Instant::now();
    for _ in 0..100 {
        let _output = bare_bones_corr(v.clone(), w);
    }
    dbg!(start_time.elapsed());
    println!("library code");
    let p = normal_vec(100000);
    let q = normal_vec(100000);
    let pq: Array2<f64> = stack![Axis(0), p, q].into_shape((2, 100000)).unwrap();

    let next = Instant::now();
    for _ in 0..100 {
        let _output = pq.pearson_correlation().unwrap();
    }
    dbg!(next.elapsed());
}

fn bare_bones_corr(v: Array1<f64>, w: Array1<f64>) -> f64 {
    let v_tilde = v.clone() - v.mean().unwrap();
    let w_tilde = w.clone() - w.mean().unwrap();
    let corr = (&v_tilde.dot(&w_tilde)) / (l2_norm(v_tilde) * l2_norm(w_tilde));
    corr
}
