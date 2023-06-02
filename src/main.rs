use std::time::Instant;

use ndarray::arr2;

use crate::matrix::trace;

mod matrix;
mod vec;

fn main() {
    let start_time = Instant::now();

    let a = matrix::bm_mat(100);

    let at = &a.t();
    let ata = at.dot(&a);
    let trace_ata = trace(ata).sqrt();
    dbg!(trace_ata);

    let at = &a.t();
    let aat = &a.dot(at);
    let trace_aat = trace(aat.clone()).sqrt();
    dbg!(trace_aat);

    dbg!(start_time.elapsed());
}
