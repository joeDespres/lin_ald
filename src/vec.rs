use ndarray::prelude::*;

pub fn l2_norm(x: Array1<f64>) -> f64 {
    x.dot(&x).sqrt()
}
pub fn unit_vec(x: Array1<f64>) -> Array1<f64> {
    x.clone() / l2_norm(x)
}
#[test]
fn testing_l2_norm() {
    let norm = l2_norm(arr1(&[1., 2., 2.]));
    assert_eq!(norm, 3.);
}
#[test]
fn test_unit_vec() {
    let x = arr1(&[3.0, 4.0, 3.0]);
    let expected = arr1(&[0.5144957554275265, 0.6859943405700353, 0.5144957554275265]);
    let result = unit_vec(x);
    assert_eq!(result, expected);
}
#[test]
fn relative_eq() {
    approx::assert_relative_eq!(1.0, 1.0, epsilon = f64::EPSILON);
    approx::assert_relative_eq!(1.0, 2.0, max_relative = 1.0);
    approx::assert_relative_eq!(1.0, 1.001, epsilon = 1e-3);
}
