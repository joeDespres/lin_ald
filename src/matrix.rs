use crate::vec;
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
#[test]
fn test_communitivity() {
    for i in 1..100 {
        let d = i;
        let a = Array::from_shape_vec((d, d), vec::gen_brownian_motion(d * d).to_vec()).unwrap();
        let b = Array::from_shape_vec((d, d), vec::gen_brownian_motion(d * d).to_vec()).unwrap();
        show_communitivity(a.clone(), b.clone(), 0.2);
    }
}

#[allow(dead_code)]
fn show_communitivity(a: Array2<f64>, b: Array2<f64>, sigma: f64) {
    let r1 = sigma * (a.clone() + b.clone());
    let r2 = sigma * a.clone() + sigma * b.clone();
    let r3 = a.clone() * sigma + b.clone() * sigma;
    let diff1 = r1.clone() - r2.clone();
    let diff2 = r1.clone() - r3.clone();
    let diff3 = r2.clone() - r3.clone();

    assert!(diff1.max().abs() < vec::TOL);
    assert!(diff2.max().abs() < vec::TOL);
    assert!(diff3.max().abs() < vec::TOL);
}

pub trait MyMatrixMethods<T>
where
    T: ndarray::NdFloat,
{
    fn max(&self) -> T;
    fn min(&self) -> T;
}

impl<T> MyMatrixMethods<T> for Array2<T>
where
    T: ndarray::NdFloat,
{
    fn max(&self) -> T {
        let mut vec = self.clone().into_raw_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *vec.first().unwrap()
    }
    fn min(&self) -> T {
        let mut vec = self.clone().into_raw_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *vec.last().unwrap()
    }
}

#[allow(dead_code)]
pub fn mat_mul(a: Array2<f64>, b: Array2<f64>) -> Array2<f64> {
    assert_eq!(a.ncols(), b.nrows());
    let mut c: Array2<f64> = Array::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            for k in 0..a.ncols() {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    c
}
#[test]
fn test_mat_mul() {
    let d = 50;
    let p = 150;
    let a = Array::from_shape_vec((d, p), vec::gen_brownian_motion(d * p).to_vec()).unwrap();
    let b = Array::from_shape_vec((p, d), vec::gen_brownian_motion(d * p).to_vec()).unwrap();
    let c = mat_mul(a.clone(), b.clone());
    let e = a.dot(&b);
    let diff = c - e;
    dbg!(&diff.max());
    assert!(diff.max().abs() < 1e-6);
}

#[allow(dead_code)]
pub fn g(d: usize, p: usize) -> Result<Array2<f64>, ndarray::ShapeError> {
    Array2::from_shape_vec((d, p), vec::gen_brownian_motion(d * p).to_vec())
}

#[test]
fn test_live_evil() {
    let l = g(2, 6).unwrap();
    let i = g(6, 3).unwrap();
    let v = g(3, 5).unwrap();
    let e = g(5, 2).unwrap();

    let live = l.dot(&i).dot(&v).dot(&e);
    let out = live.t();
    let other = e.t().dot(&v.t()).dot(&i.t()).dot(&l.t());

    let diff = &out - other;
    assert!(diff.max() < vec::TOL);
}
#[allow(dead_code)]
pub fn is_symmetric(a: Array2<f64>) -> bool {
    if &a.nrows() != &a.ncols() {
        return false;
    }
    let a_t = &a.t();
    let diff = a_t - &a;
    diff.min() < vec::TOL
}
#[test]
fn test_is_symetrix() {
    let m = array![[1., 2.], [2., 1.]];
    assert!(is_symmetric(m));

    let m = g(5, 5).unwrap();

    assert!(!is_symmetric(m));
}
