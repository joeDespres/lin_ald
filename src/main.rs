use approx::assert_relative_eq;
use ndarray::prelude::*;
use std::time::Instant;

use crate::vec::gen_brownian_motion;
mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    let d = 2;
    let a = Array::from_shape_vec((d, d), gen_brownian_motion(d * d).to_vec()).unwrap();
    let b = Array::from_shape_vec((d, d), gen_brownian_motion(d * d).to_vec()).unwrap();

    show_communitivity(a.clone(), b.clone(), 0.2);
    dbg!(a.max());

    dbg!(start_time.elapsed());
}

fn show_communitivity(a: Array2<f64>, b: Array2<f64>, sigma: f64) {
    let r1 = sigma * (a.clone() + b.clone());
    let r2 = sigma * a.clone() + sigma * b.clone();
    let r3 = a.clone() * sigma + b.clone() * sigma;
    let diff1 = r1.clone() - r2.clone();
    let diff2 = r1.clone() - r3.clone();
    let diff3 = r2.clone() - r3.clone();
    // assert!(diff1.max().unwrap() < vec::TOL);
    // assert!(diff2.max().unwrap() < vec::TOL);
    // assert!(diff3.max().unwrap() < vec::TOL);
}

pub trait MyMatrixMethods<T>
where
    T: ndarray::NdFloat,
{
    fn max(&self) -> T;
}

impl<T> MyMatrixMethods<T> for Array1<T>
where
    T: ndarray::NdFloat,
{
    fn max(&self) -> T {
        self.clone()
            .into_raw_vec()
            .sort_by(|a, b| a.partial_cmp(b).unwrap())
            .last()
    }
}
