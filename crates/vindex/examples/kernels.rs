//! Time the distance kernels on data that sits in the chip's cache: the arithmetic cost alone, per item.
//! cargo run --release -p vindex --example kernels
use std::hint::black_box;
use std::time::Instant;

use vindex::codes::{dot_u8, dot_u8_n};
use vindex::metric::{dot, dot_n};

fn main() {
    for d in [128usize, 200, 768] {
        let a: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin()).collect();
        let rows: Vec<Vec<u8>> = (0..64).map(|r| (0..d).map(|i| ((i * 7 + r * 13) % 256) as u8).collect()).collect();
        let full: Vec<Vec<f32>> = (0..64).map(|r| (0..d).map(|i| ((i * 3 + r) as f32 * 0.11).cos()).collect()).collect();
        let reps = 400_000;
        let time = |f: &mut dyn FnMut(usize) -> f32, per: usize| {
            let t = Instant::now();
            let mut s = 0f32;
            for k in 0..reps / per {
                s += f(k);
            }
            black_box(s);
            t.elapsed().as_nanos() as f64 / reps as f64
        };
        let one = time(&mut |k| dot_u8(black_box(&a), &rows[k % 64]), 1);
        let four = time(&mut |k| {
            let b = (k * 4) % 64;
            dot_u8_n(black_box(&a), [&rows[b], &rows[b + 1], &rows[b + 2], &rows[b + 3]]).iter().sum()
        }, 4);
        let f_one = time(&mut |k| dot(black_box(&a), &full[k % 64]), 1);
        let f_four = time(&mut |k| {
            let b = (k * 4) % 64;
            dot_n(black_box(&a), [&full[b], &full[b + 1], &full[b + 2], &full[b + 3]]).iter().sum()
        }, 4);
        println!("{d:4} numbers: 1-byte copy {one:5.1} ns alone, {four:5.1} ns four at once | full numbers {f_one:5.1} ns alone, {f_four:5.1} ns four at once");
    }
}
