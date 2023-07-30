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
    use crate::matrix::MyMatrixMethods;
    use crate::TOL;
    use ndarray::arr2;
    let w = arr2(&[
        [1., 1., 1., 3., 1., 8., 6., 9.],
        [1., 3., 2., 3., 1., 7., 8., 3.],
        [2., 0., 1., 3., 1., 5., 3., 6.],
    ]);
    let r = w.right_inverse();
    let wr = w.dot(&r);
    let target: Array2<f64> = Array2::eye(3);
    dbg!((wr - target).abs_max());
}
