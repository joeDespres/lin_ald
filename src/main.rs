mod lin_ops;
mod matrix;
mod vec;
use crate::lin_ops::MyLinOps;
use crate::matrix::bm_mat;
use crate::matrix::hilbert_mat;
use crate::matrix::MyMatrixMethods;
use crate::vec::TOL;
use ndarray::Array;
use ndarray::Array2;
use ndarray_linalg::Inverse;
use std::time::Instant;
fn main() {
    use crate::matrix::MyMatrixMethods;
    use ndarray::Array1;
    let h = hilbert_mat(70000.0);
    dbg!(h);
}
