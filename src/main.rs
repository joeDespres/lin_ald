mod lin_ops;
mod matrix;
mod vec;
use crate::lin_ops::MyLinOps;
use crate::matrix::bm_mat;
use crate::matrix::MyMatrixMethods;
use crate::vec::TOL;
use ndarray::Array2;
use ndarray_linalg::Inverse;
use std::time::Instant;

fn main() {
    for i in 2.. {
        let a = bm_mat(i);
        let start_time = Instant::now();
        let a_inv = a.inv().unwrap();
        let elapsed_time = start_time.elapsed();
        println!("Elapsed time ndarray: {:?}", elapsed_time);
        let start_time = Instant::now();
        let amyinv = a.invert();
        let elapsed_time = start_time.elapsed();
        println!("Elapsed time mine: {:?}", elapsed_time);
        let diff = (&a_inv - amyinv).abs_max();
        dbg!(diff);
        dbg!(i);
        assert!(diff < TOL);
    }
}
