use ndarray::{Array, Array2, Axis};
use ndarray_linalg::Determinant;
use ndarray_linalg::Inverse;
use num::One;

pub trait MyLinOps<T>
where
    T: ndarray::NdFloat,
{
    fn minors_mat(&self) -> Array2<T>;
    fn sub_matricies(&self, rm_rows: &[usize], rm_cols: &[usize]) -> Array2<T>;
    fn grid_mat(&self) -> Array2<T>;
    fn cofactors_mat(&self) -> Array2<T>;
    fn invert(&self) -> Array2<T>;
    fn left_inverse(&self) -> Array2<T>;
}

impl<T> MyLinOps<T> for Array2<T>
where
    T: ndarray::NdFloat + Clone + ndarray_linalg::Lapack,
{
    fn left_inverse(&self) -> Array2<T> {
        let tall_mat = self;
        let tall_mat_tall_mat_transpose = tall_mat.t().to_owned().dot(tall_mat);
        let t_t_t_inv = tall_mat_tall_mat_transpose.inv().unwrap();
        let l = t_t_t_inv.dot(&self.t().to_owned());
        return l;
    }

    fn invert(&self) -> Array2<T> {
        let minors_mat = self.minors_mat();
        let cofactors = minors_mat.cofactors_mat();
        let cofactors_t = cofactors.t().to_owned();
        let scale = self.det().unwrap();

        let nrow = self.nrows();
        let ncol = self.ncols();
        let scale_mat = Array2::from_elem((nrow, ncol), scale);
        cofactors_t / scale_mat
    }

    fn cofactors_mat(&self) -> Array2<T> {
        self * self.grid_mat()
    }

    fn grid_mat(&self) -> Array2<T> {
        let n_rows = self.nrows();
        let n_cols = self.ncols();
        assert_eq!(n_rows, n_cols);
        let mut row_iter = n_rows;
        let mut col_iter = n_cols;

        if n_rows % 2 == 0 {
            row_iter += 1;
            col_iter += 1;
        }

        let mut grid_mat: Array2<T> = Self::zeros((row_iter, col_iter));
        let mut iter = 0;
        for row_i in 0..row_iter {
            for col_j in 0..col_iter {
                if iter % 2 == 0 {
                    grid_mat[[row_i, col_j]] += One::one();
                } else {
                    grid_mat[[row_i, col_j]] -= One::one();
                }
                iter += 1;
            }
        }
        grid_mat.slice(ndarray::s![..n_rows, ..n_cols]).to_owned()
    }

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
fn test_left_inverse() {
    use crate::matrix::MyMatrixMethods;
    use crate::TOL;
    use ndarray::arr2;
    let t = arr2(&[
        [1., 1., 1.],
        [7., 3., 1.],
        [0., 3., 1.],
        [0., 9., 1.],
        [1., 0., 2.],
        [0., 3., 1.],
        [2., 3., 2.],
    ]);
    let l = t.left_inverse();
    let lt = l.dot(&t);
    let target: Array2<f64> = Array2::eye(3);
    assert!((lt - target).abs_max() < TOL);
}

#[test]
fn test_grid() {
    use ndarray::arr2;
    let a = arr2(&[[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]]);
    let grid = a.grid_mat();
    let target = arr2(&[[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]]);
    assert_eq!(grid, target);
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let g = a.grid_mat();
    let target = arr2(&[[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]);
    assert_eq!(g, target);

    let a = arr2(&[
        [1., 1., 1., 0.],
        [0., 3., 1., 2.],
        [1., 0., 2., 1.],
        [2., 3., 1., 0.],
    ]);
    let grid = a.grid_mat();
    let target = arr2(&[
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ]);
    assert_eq!(grid, target);
}
#[test]
fn nest_invert() {
    use crate::matrix::bm_mat;
    use crate::matrix::MyMatrixMethods;
    use crate::vec::TOL;
    use ndarray::arr2;

    let a = arr2(&[
        [1., 1., 1., 0.],
        [0., 3., 1., 2.],
        [1., 0., 2., 1.],
        [2., 3., 1., 0.],
    ]);

    let a_inv = a.invert();
    let target = arr2(&[
        [-3.0, 1.00, 3.00, -3.],
        [-0.5, 0.25, 0.25, 0.],
        [1.0, -0.50, -0.50, 1.],
        [1.5, -0.25, -1.25, 1.],
    ])
    .t()
    .to_owned();

    let diff = (a_inv - target).abs_max();
    assert!(diff < TOL);
    let a = arr2(&[[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]]);
    let a_inv = a.invert();
    let target = arr2(&[[-24., 20., -5.], [18., -15., 4.], [5., -4., 1.]])
        .t()
        .to_owned();
    let diff = (a_inv - target).abs_max();
    assert!(diff < TOL);

    let a = arr2(&[[0., 1., 1.], [2., 2., 2.], [2., 6., 1.]]);
    let a_inv = a.invert();
    let target = arr2(&[[-1., 0.2, 0.8], [0.5, -0.2, 0.2], [0., 0.2, -0.2]])
        .t()
        .to_owned();
    let diff = (a_inv - target).abs_max();
    assert!(diff < TOL);

    let a = bm_mat(10);
    let a_inv = a.invert();
    let target: Array2<f64> = Array2::eye(10);
    let aainv = a.dot(&a_inv);
    let diff = (&aainv - &target).abs_max();
    assert!(diff < TOL);
}
#[test]
fn test_cofactors_mat() {
    use ndarray::arr2;
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let c = a.cofactors_mat();
    let target = arr2(&[[1., -2., 3.], [-4., 5., -6.], [7., -8., 9.]]);
    assert_eq!(c, target);
}

#[test]
fn test_minors_mat() {
    use crate::matrix::MyMatrixMethods;
    use crate::vec::TOL;
    use ndarray::arr2;
    let a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let m = a.minors_mat();
    let target = arr2(&[[-3., -6., -3.], [-6., -12., -6.], [-3., -6., -3.]]);
    assert!((m - target).abs_max() < TOL);
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
