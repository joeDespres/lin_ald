use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

#[allow(dead_code)]
pub fn l2_norm(x: Array1<f64>) -> f64 {
    x.dot(&x).sqrt()
}
#[allow(dead_code)]
pub fn unit_vec(x: Array1<f64>) -> Array1<f64> {
    x.clone() / l2_norm(x)
}
#[allow(dead_code)]
pub fn normal_vec(n: usize) -> Array1<f64> {
    Array::random(n, Normal::new(0., 1.).unwrap())
}
#[allow(dead_code)]
pub fn set_vec_len(mag: f64, vec: Array1<f64>) -> Array1<f64> {
    unit_vec(vec) * mag
}
#[test]
fn test_set_vec_len() {
    let w = normal_vec(69);
    let v = set_vec_len(9., w);
    let len_v = l2_norm(v);
    approx::assert_relative_eq!(len_v, 9.0, epsilon = 1e-10);
    approx::assert_relative_eq!(v.len(), 9.0, epsilon = 1e-10);
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
    for _ in 1..1_000 {
        let v = normal_vec(69);
        let unit_vec_v = unit_vec(v);
        let length = unit_vec_v.mapv(|x| x.powi(2)).sum();
        approx::assert_relative_eq!(length, 1.0, epsilon = 1e-12);
    }
}
#[test]
fn relative_eq() {
    approx::assert_relative_eq!(1.0, 1.0, epsilon = f64::EPSILON);
    approx::assert_relative_eq!(1.0, 2.0, max_relative = 1.0);
    approx::assert_relative_eq!(1.0, 1.001, epsilon = 1e-3);
}
