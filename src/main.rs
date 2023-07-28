mod lin_ops;
mod matrix;
mod vec;
use crate::lin_ops::MyLinOps;
use crate::matrix::bm_mat;
use crate::matrix::MyMatrixMethods;
use crate::vec::TOL;
use ndarray::arr2;
use ndarray_linalg::Inverse;
fn main() {
    let a = arr2(&[
        [1., 1., 1., 0.],
        [0., 3., 1., 2.],
        [1., 0., 2., 1.],
        [2., 3., 1., 0.],
    ]);

    let a_inv = &a.invert();
    dbg!(&a_inv);

    let target = arr2(&[
        [-3.0, 1.00, 3.00, -3.],
        [-0.5, 0.25, 0.25, 0.],
        [1.0, -0.50, -0.50, 1.],
        [1.5, -0.25, -1.25, 1.],
    ]);
    let diff = (a_inv - target).abs_max();
    dbg!(diff);
    assert!(diff < TOL);
}
