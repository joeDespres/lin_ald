use crate::vec;
use ndarray::{concatenate, prelude::*};
use ndarray_linalg::{into_col, into_row};

#[allow(dead_code)]
pub fn acess_indx(a: Array2<f64>, i: usize, j: usize) -> String {
    format!("the matrix element at a[[{}, {}]] is {}", i, j, a[[i, j]])
}
#[allow(dead_code)]
pub fn hilbert_mat(c: f64) -> Array2<f64> {
    let a: Array1<f64> = Array::range(1., c + 1., 1.);
    let a = into_col(a);
    let b: Array1<f64> = Array::range(0., c, 1.);
    let b = into_row(b);
    (a + b).mapv(|x| x.powi(-1))
}

#[test]
fn test_hilbert_mat() {
    let h = hilbert_mat(3.0);
    let target = arr2(&[
        [1., 1. / 2., 1. / 3.],
        [1. / 2., 1. / 3., 1. / 4.],
        [1. / 3., 1. / 4., 1. / 5.],
    ]);
    assert_eq!(h, target);
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
    let r3 = a * sigma + b * sigma;
    let diff1 = r1.clone() - r2.clone();
    let diff2 = r1 - r3.clone();
    let diff3 = r2 - r3;

    assert!(diff1.abs_max().abs() < vec::TOL);
    assert!(diff2.abs_max().abs() < vec::TOL);
    assert!(diff3.abs_max().abs() < vec::TOL);
}

pub trait MyMatrixMethods<T>
where
    T: ndarray::NdFloat,
{
    fn max(&self) -> T;
    fn min(&self) -> T;
    fn abs_max(&self) -> T;
}

impl<T> MyMatrixMethods<T> for Array2<T>
where
    T: ndarray::NdFloat + num::Signed,
{
    fn max(&self) -> T {
        let mut vec = self.clone().into_raw_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *vec.last().unwrap()
    }
    fn min(&self) -> T {
        let mut vec = self.clone().into_raw_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *vec.first().unwrap()
    }
    fn abs_max(&self) -> T {
        let mut vec = self.clone().into_raw_vec();
        vec.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        let larges_absolute_value = *vec.last().unwrap();
        return num::abs(larges_absolute_value);
    }
}

#[test]
fn test_abs_max() {
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    assert_eq!(a.max(), 9.0);
    assert_eq!(a.min(), 1.0);
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, -18.0, 9.0]]);
    assert_eq!(a.abs_max(), 18.0);
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
    dbg!(&diff.abs_max());
    assert!(diff.abs_max().abs() < 1e-6);
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
    assert!(diff.abs_max() < vec::TOL);
}
#[allow(dead_code)]
pub fn is_symmetric(a: Array2<f64>) -> bool {
    if &a.nrows() != &a.ncols() {
        return false;
    }
    let a_t = &a.t();
    let diff = a_t - &a;
    dbg!(&a);
    dbg!(&a_t);
    dbg!(&diff);

    diff.abs_max() < vec::TOL
}
#[test]
fn test_is_symetrix() {
    let m = array![[1., 2.], [2., 1.]];
    assert!(is_symmetric(m));

    let m = array![[2., 3.], [2., 1.]];
    assert!(!is_symmetric(m));
}
#[allow(dead_code)]
pub fn jankey_avg(a: Array2<f64>) -> Array2<f64> {
    let a_t = &a.t();
    (a_t + &a) / 2.
}
#[test]
fn test_jankey_avg() {
    let d = 5;
    let a = Array::from_shape_vec((d, d), vec::gen_brownian_motion(d * d).to_vec()).unwrap();
    let symetric = jankey_avg(a);

    assert!(is_symmetric(symetric));
}
#[test]
fn test_hadamard_mul() {
    let d = 50;
    let a = Array2::from_diag(&arr1(&vec::gen_brownian_motion(d).to_vec()));
    let b = Array2::from_diag(&arr1(&vec::gen_brownian_motion(d).to_vec()));

    let hadamard = &a * &b;
    let dot_prod = &a.dot(&b);
    let diff = hadamard - dot_prod;
    assert!(diff.abs_max() < vec::TOL);
}
#[allow(dead_code)]
pub fn bm_mat(d: usize) -> Array2<f64> {
    Array2::from_shape_vec((d, d), vec::gen_brownian_motion(d * d).to_vec()).unwrap()
}
#[allow(dead_code)]
pub fn frobenius_norm(a: Array2<f64>) -> f64 {
    a.map(|a| a.powf(2.0)).sum().sqrt()
}

#[test]
fn test_frobenius_norm() {
    let a = arr2(&[[2., 3., 4.], [4., 5., 6.]]);
    let norm = frobenius_norm(a);
    assert!(norm - 24.0 < vec::TOL);
}
#[allow(dead_code)]
pub fn frobenius_norm_to_zero(a: Array2<f64>, b: Array2<f64>) -> f64 {
    let mut norm = 0.0;
    let mut s = 1.0;
    loop {
        let _ = norm; // compiler warning hack
        let c = s * a.clone() - s * b.clone();
        norm = frobenius_norm(c);
        if norm < 1.0 {
            break;
        }
        s *= 0.99;
    }
    norm
}

#[test]
fn test_frobenius_norm_to_zero() {
    let a = bm_mat(10);
    let b = bm_mat(10);

    let norm = frobenius_norm_to_zero(a, b);
    assert!(norm - 1.0 < vec::TOL);
}

#[allow(dead_code)]
pub fn trace(a: Array2<f64>) -> f64 {
    a.diag().sum()
}
#[test]
fn test_trace() {
    let a = bm_mat(100);
    let f = frobenius_norm(a.clone());

    let at = &a.t();
    let ata = at.dot(&a);
    let trace_ata = trace(ata).sqrt();

    assert!(f - trace_ata < vec::TOL);

    let at = &a.t();
    let aat = &a.dot(at);
    let trace_aat = trace(aat.clone()).sqrt();
    assert!(trace_ata - trace_aat < vec::TOL);
}
#[allow(dead_code)]
pub fn bm_mat_rec(i: usize, j: usize) -> Array2<f64> {
    Array2::from_shape_vec(
        (i, j),
        vec::gen_brownian_motion((i * j).try_into().unwrap()).to_vec(),
    )
    .unwrap()
}
#[test]
fn test_bm_mat_rec() {
    let rows = 420;
    let cols = 69;

    let a = bm_mat_rec(rows, cols);
    let dim = a.dim();
    assert_eq!(dim, (420, 69));
}
#[allow(dead_code)]
pub fn cov_mat(a: Array2<f64>) -> Array2<f64> {
    let n = a.nrows() as f64;
    a.t().dot(&a) / (n - 1.0)
}
#[test]
fn test_cov_mat() {
    let a = arr2(&[[-1., 0.], [0., -1.], [1., 1.]]);
    let c = cov_mat(a);
    let target = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    assert_eq!(c, target);
}
#[allow(dead_code)]
pub fn cov_to_corr(c: Array2<f64>) -> Array2<f64> {
    // todo demean the incomming mat
    // let mean_vec = c.mean_axis(Axis(0)).unwrap();
    // let d = c.clone() - mean_vec;
    let s = c.diag().map(|a| a.sqrt().powi(-1));
    let s_mat = Array2::from_diag(&s);
    s_mat.dot(&c).dot(&s_mat)
}
#[test]
fn test_cov_to_corr() {
    let c = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let target = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
    let r = cov_to_corr(c);
    assert_eq!(target, r);
}
