use ndarray::prelude::*;
use std::time::Instant;

mod vec;

fn main() {
    let start_time = Instant::now();
    let w = vec::normal_vec(69);
    let v = vec::set_vec_len(9., w);
    dbg!(v.len());

    let a = arr1(&[1.0, 2.]);
    let len = vec::l2_norm(a);
    dbg!(len);

    let len_v = vec::l2_norm(v);

    dbg!(start_time.elapsed());
}
