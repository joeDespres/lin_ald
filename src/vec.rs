use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

#[allow(dead_code)]
pub fn l2_norm(x: Array1<f64>) -> f64 {
    x.dot(&x).sqrt()
}
#[test]
fn testing_l2_norm() {
    let norm = l2_norm(arr1(&[1., 2., 2.]));
    assert_eq!(norm, 3.);
}
#[allow(dead_code)]
pub fn unit_vec(x: Array1<f64>) -> Array1<f64> {
    x.clone() / l2_norm(x)
}
#[test]
fn test_unit_vec() {
    let x = arr1(&[3.0, 4.0, 3.0]);
    let expected = arr1(&[0.514_495_755_427_526_5, 0.685_994_340_570_035_3, 0.514_495_755_427_526_5]);
    let result = unit_vec(x);
    assert_eq!(result, expected);
    for _ in 1..1_000 {
        let v = normal_vec(69);
        let unit_vec_v = unit_vec(v);
        let length = unit_vec_v.mapv(|x| x.powi(2)).sum();
        approx::assert_relative_eq!(length, 1.0, epsilon = 1e-12);
    }
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
}
#[allow(dead_code)]
pub fn reorient_vec(v: Array2<f64>) -> Array2<f64> {
    let dim_v = v.dim();
    let new_dims = (dim_v.1, dim_v.0);
    let mut output = Array::zeros((dim_v.1, dim_v.0));
    for column in 0..new_dims.0 {
        for row in 0..new_dims.1 {
            output[(column, row)] = v[(row, column)];
        }
    }
    output
}
#[test]
fn test_reorient_vector() {
    let v = Array::from_elem((1, 4), 69.);
    let w = reorient_vec(v);
    assert_eq!(w.dim(), (4, 1));

    let v = Array::from_elem((14, 89), 69.);
    let w = reorient_vec(v);
    assert_eq!(w.dim(), (89, 14));
}
#[test]
fn dot_prod_communitive() {
    let v = arr1(&[4, 5, 3]);
    let w = arr1(&[5, 4, 6]);
    assert_eq!(&v.dot(&w), &w.dot(&v));
}
pub fn orth_decomp(a: &Array1<f64>, b: Array1<f64>) -> f64 {
    a.dot(&b) / a.dot(a)
}
#[test]
fn test_orth_decomp() {
    let v = arr1(&[4., 5., 3.]);
    let w = arr1(&[5., 4., 6.]);
    let beta = orth_decomp(&v, w);
    assert_eq!(beta, 1.16);
}
#[allow(dead_code)]
pub fn parallel_component(a: Array1<f64>, b: Array1<f64>) -> Array1<f64> {
    let beta = orth_decomp(&a, b);
    a * beta
}
#[test]
fn test_parallel_components() {
    let n: usize = 2000;
    let a = normal_vec(n);
    let b = normal_vec(n);

    let a_parallel = parallel_component(a.clone(), b.clone());
    let a_orth = &b - &a_parallel;
    approx::assert_relative_eq!(a.dot(&a_orth), 0., epsilon = 1e-10);
}

#[allow(dead_code)]
pub struct OrthogonalDecomp {
    parallel_component: Array1<f64>,
    orthogonal_component: Array1<f64>,
}

#[allow(dead_code)]
pub fn orthogonal_decomp(a: &Array1<f64>, b: &Array1<f64>) -> OrthogonalDecomp {
    let beta = a.dot(b) / a.dot(a);
    let parallel_component = a * beta;
    let orthogonal_component = b - &parallel_component;
    
    OrthogonalDecomp {
        parallel_component,
        orthogonal_component,
    }
}
#[test]
fn test_orth_decomp_struct() {
    let n: usize = 2_000_000;
    let a = normal_vec(n);
    let b = normal_vec(n);
    let orth_decomp = orthogonal_decomp(&a, &b);
    approx::assert_relative_eq!(
        orth_decomp
            .orthogonal_component
            .dot(&orth_decomp.parallel_component),
        0.,
        epsilon = 1e-10
    );
}
