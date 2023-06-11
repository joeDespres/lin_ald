use ndarray::Array2;
use ndarray_linalg::Determinant;

pub trait MyLinOps<T>
where
    T: ndarray::NdFloat,
{
    fn minors_mat(&self) -> Array2<T>;
}

impl<T> MyLinOps<T> for Array2<T>
where
    T: ndarray::NdFloat,
{
    fn minors_mat(&self) -> Array2<T> {
        let n_rows = self.nrows();
        let n_cols = self.ncols();
        dbg!(n_rows);
        dbg!(n_cols);

        self.clone()
    }
}
