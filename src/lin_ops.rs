use ndarray::Array;
use ndarray::Array2;
use ndarray::Axis;
// use ndarray_linalg::Determinant;

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

        dbg!(&self);
        let mut iter = 0;
        for row_i in 0..n_rows {
            for col_j in 0..n_cols {
                iter += 1;
                let a = remove_rows(self, &[row_i]);
                let b = remove_cols(&a, &[col_j]);
                dbg!(b);

                // dbg!(sliced_mat);
            }
        }

        self.clone()
    }
}

fn remove_rows<A: Clone>(matrix: &Array2<A>, to_remove: &[usize]) -> Array2<A> {
    let mut keep_row = vec![true; matrix.nrows()];
    to_remove.iter().for_each(|row| keep_row[*row] = false);

    let elements_iter = matrix
        .axis_iter(Axis(0))
        .zip(keep_row.iter())
        .filter(|(_row, keep)| **keep)
        .flat_map(|(row, _keep)| row.to_vec());

    let new_n_rows = matrix.nrows() - to_remove.len();
    Array::from_iter(elements_iter)
        .into_shape((new_n_rows, matrix.ncols()))
        .unwrap()
}
fn remove_cols<A: Clone>(matrix: &Array2<A>, to_remove: &[usize]) -> Array2<A> {
    let mut keep_col = vec![true; matrix.ncols()];
    to_remove.iter().for_each(|row| keep_col[*row] = false);

    let elements_iter = matrix
        .axis_iter(Axis(1))
        .zip(keep_col.iter())
        .filter(|(_col, keep)| **keep)
        .flat_map(|(col, _keep)| col.to_vec());

    let new_n_cols = matrix.ncols() - to_remove.len();
    Array::from_iter(elements_iter)
        .into_shape((matrix.nrows(), new_n_cols))
        .unwrap()
}
