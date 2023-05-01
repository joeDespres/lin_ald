use ndarray::prelude::*;
use std::time::Instant;
use vec::normal_vec;

mod vec;

fn main() {
    let start_time = Instant::now();
    let v = gen_brownian_motion(69);
    dbg!(&v);
    let kernel = arr1(&[-1.0, 1.0]);
    let output = v.kernel_mul(&kernel);
    dbg!(output);

    dbg!(start_time.elapsed());
}

fn gen_brownian_motion(n: usize) -> Array1<f64> {
    let mut n = normal_vec(n);
    n.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    n
}

trait MyVecMethods<T>
where
    T: ndarray::LinalgScalar,
{
    fn kernel_mul(&self, kernel: &Array1<T>) -> Array1<T>;
}

impl<T> MyVecMethods<T> for Array1<T>
where
    T: ndarray::LinalgScalar + std::ops::Mul<Output = T> + std::ops::AddAssign,
{
    fn kernel_mul(&self, kernel: &Array1<T>) -> Array1<T> {
        let l = kernel.len();

        self.windows(l)
            .into_iter()
            .map(|x| x.dot(kernel))
            .collect::<Vec<T>>()
            .into()
    }
}
