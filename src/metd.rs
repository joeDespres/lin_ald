use ndarray::prelude::*;
pub trait MyVecMethods<T>
where
    T: ndarray::LinalgScalar,
{
    fn kernel_mul(&self, kernel: &Array1<T>) -> Array1<T>;
}

impl<T> MyVecMethods<T> for Array1<T>
where
    T: ndarray::LinalgScalar,
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

pub trait FloatMethods<T>
where
    T: ndarray::NdFloat,
{
    fn l2_norm(&self) -> T;
    fn unit_vec(&self) -> Array1<T>;
    fn set_magnitude(&self, l: T) -> Array1<T>;
}

impl<T> FloatMethods<T> for Array1<T>
where
    T: ndarray::NdFloat,
{
    fn l2_norm(&self) -> T {
        self.dot(self).sqrt()
    }
    fn unit_vec(&self) -> Array1<T> {
        self / self.l2_norm()
    }
    fn set_magnitude(&self, l: T) -> Array1<T> {
        self.unit_vec() * l
    }
}

#[test]
fn testing_l2_norm() {
    use vec::TOL;
    let norm = arr1(&[1., 2., 2.]).l2_norm();
    approx::assert_relative_eq!(norm, 3., epsilon = TOL);
}
#[test]
fn test_unit_vec() {
    use vec::TOL;
    let x = arr1(&[3.0, 4.0, 3.0]);
    let expected = arr1(&[
        0.514_495_755_427_526_5,
        0.685_994_340_570_035_3,
        0.514_495_755_427_526_5,
    ]);
    let result = x.unit_vec();
    assert_eq!(result, expected);
    for _ in 1..1_000 {
        let v = vec::normal_vec(69);
        let unit_vec_v = v.unit_vec();
        let length = unit_vec_v.mapv(|x| x.powi(2)).sum();
        approx::assert_relative_eq!(length, 1.0, epsilon = TOL);
    }
}
#[test]
fn test_set_vec_len() {
    use vec::TOL;
    let len_v = vec::normal_vec(69).set_magnitude(9.0).l2_norm();
    approx::assert_relative_eq!(len_v, 9.0, epsilon = TOL);
}
