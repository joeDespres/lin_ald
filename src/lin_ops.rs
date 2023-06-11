use ndarray::{Array, Array2, Axis};
use ndarray_linalg::Determinant;

pub trait MyLinOps<T>
where
    T: ndarray::NdFloat,
{
    fn minors_mat(&self) -> Array2<T>;
    fn sub_matricies(&self, rm_rows: &[usize], rm_cols: &[usize]) -> Array2<T>;
}

impl<T> MyLinOps<T> for Array2<T>
where
    T: ndarray::NdFloat + Clone + ndarray_linalg::Lapack,
{
    fn minors_mat(&self) -> Self {
        let n_rows = self.nrows();
        let n_cols = self.ncols();

        let mut minors_mat = Self::zeros((n_rows, n_cols));
        for row_i in 0..n_rows {
            for col_j in 0..n_cols {
                let s = self.sub_matricies(&[row_i], &[col_j]);
                let determinent = Determinant::det(&s).unwrap();
                minors_mat[[row_i, col_j]] = determinent;
            }
        }

        minors_mat
    }
    fn sub_matricies(&self, rm_rows: &[usize], rm_cols: &[usize]) -> Self {
        let mut keep_row = vec![true; self.nrows()];
        rm_rows.iter().for_each(|row| keep_row[*row] = false);

        let elements_iter = self
            .axis_iter(Axis(0))
            .zip(keep_row.iter())
            .filter(|(_row, keep)| **keep)
            .flat_map(|(row, _keep)| row.to_vec());

        let new_n_rows = self.nrows() - rm_rows.len();
        let mat_without_rows = Array::from_iter(elements_iter)
            .into_shape((new_n_rows, self.ncols()))
            .unwrap();

        let mut keep_col = vec![true; mat_without_rows.ncols()];
        rm_cols.iter().for_each(|row| keep_col[*row] = false);

        let elements_iter = mat_without_rows
            .axis_iter(Axis(1))
            .zip(keep_col.iter())
            .filter(|(_col, keep)| **keep)
            .flat_map(|(col, _keep)| col.to_vec());

        let new_n_cols = self.ncols() - rm_cols.len();

        Array::from_iter(elements_iter)
            .into_shape((mat_without_rows.nrows(), new_n_cols))
            .unwrap()
            .reversed_axes()
    }
}
#[test]
fn test_minors_mat() {
    use crate::matrix::MyMatrixMethods;
    use crate::vec::TOL;
    use ndarray::arr2;
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m = a.minors_mat();
    let target = arr2(&[[-3., -6., -3.], [-6., -12., -6.], [-3., -6., -3.]]);

    assert!((m - target).min() < TOL);
}

#[test]
fn test_submatrixies() {
    use ndarray::arr2;
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let s = a.sub_matricies(&[0], &[0]);
    let target = arr2(&[[5., 6.], [8., 9.]]);
    assert_eq!(s, target);

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let s = a.sub_matricies(&[1], &[1]);
    let target = arr2(&[[1., 3.], [7., 9.]]);
    assert_eq!(s, target);

    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let s = a.sub_matricies(&[0], &[2]);
    let target = arr2(&[[4., 5.], [7., 8.]]);
    assert_eq!(s, target);
}
