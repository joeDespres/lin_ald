mod lin_ops;
mod matrix;
mod vec;

use ndarray::array;
fn main() {
    let m = array![[2., 3.], [2., 1.]];
    assert!(matrix::is_symmetric(m));
}
