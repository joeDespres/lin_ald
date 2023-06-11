use crate::lin_ops::MyLinOps;
use ndarray::arr2;
use std::time::Instant;
mod lin_ops;
mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    dbg!(&a.minors_mat());

    let s = a.sub_matricies(&[0], &[0]);
    let target = arr2(&[[5., 6.], [8., 9.]]);
    assert_eq!(s, target);

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let s = a.sub_matricies(&[1], &[1]);
    let target = arr2(&[[1., 3.], [7., 9.]]);
    assert_eq!(s, target);

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let s = a.sub_matricies(&[0], &[2]);
    let target = arr2(&[[4., 5.], [7., 8.]]);
    assert_eq!(s, target);
    // let det_a = a.det().unwrap();
    dbg!(start_time.elapsed());
}
