mod lin_ops;
mod matrix;
mod vec;
use crate::lin_ops::MyLinOps;
use crate::matrix::bm_mat;
use crate::matrix::MyMatrixMethods;
use ndarray_linalg::Inverse;
fn main() {
    let a = bm_mat(3);

    let i_inv = &a.inv().unwrap();
    dbg!(&i_inv);
    dbg!(&a.invert());

    let diff = (i_inv - &a.invert()).abs_max();

    dbg!(diff);
}
