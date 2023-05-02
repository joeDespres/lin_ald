use ndarray::{concatenate, prelude::*};

#[allow(dead_code)]
pub fn acess_indx(a: Array2<f64>, i: usize, j: usize) -> String {
    format!("the matrix element at a[[{}, {}]] is {}", i, j, a[[i, j]])
}

#[allow(dead_code)]
pub fn reshuffel_test(c: Array2<f64>) -> Array2<f64> {
    let c1 = c.slice(s![0..5, 0..5]);
    let c2 = c.slice(s![0..5, 5..10]);
    let c3 = c.slice(s![5..10, 0..5]);
    let c4 = c.slice(s![5..10, 5..10]);
    let c21 = concatenate![Axis(1), c2, c1];
    let c43 = concatenate![Axis(1), c4, c3];
    concatenate![Axis(0), c43, c21]
}

#[allow(dead_code)]
pub fn element_addition(a: Array2<f64>, b: Array2<f64>) -> Array2<f64> {
    assert!(a.dim() == b.dim());
    let rows = a.nrows();
    let cols = a.ncols();
    let mut c = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let cij = a[[i, j]] + b[[i, j]];
            c[[i, j]] = cij;
        }
    }
    c
}
#[test]
fn test_element_addition() {
    let v = Array::range(0., 1_000_000., 1.);
    let c0 = Array::from_shape_vec((1_000, 1_000), v.to_vec()).unwrap();
    let c1 = Array::from_shape_vec((1_000, 1_000), v.to_vec()).unwrap();
    let c = element_addition(c0, c1);
    let a = Array::range(0., 2_000_000., 2.);
    let target = Array::from_shape_vec((1_000, 1_000), a.to_vec()).unwrap();

    assert_eq!(c, target);
}
