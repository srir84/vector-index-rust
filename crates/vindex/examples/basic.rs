//! Build, search, change, save and load, from Rust.
//! cargo run --release -p vindex --example basic
use vindex::rng::Rng;
use vindex::{Effort, Index, Metric, Numbers, Settings};

fn random_rows(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..n * dim).map(|_| (r.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect()
}

fn main() -> Result<(), vindex::Error> {
    let (n, dim) = (20_000, 64);
    let items = random_rows(n, dim, 1);
    let ids: Vec<u64> = (0..n as u64).collect();
    let training = random_rows(2_000, dim, 2); // sample questions, never the ones you test with
    let questions = random_rows(100, dim, 3);

    let mut idx = Index::new(dim, Metric::Cos, Settings::default())?;
    idx.build(&items, &ids, Some(&training))?;

    // learned effort: aim for 95% of the true top 10; or Effort::List(100) for a fixed list size
    let out = idx.search(&questions, 10, Effort::Target(0.95))?;
    println!("first question's top 3: {:?}, checks {}", &out.ids[..3], out.checks[0]);

    idx.add(1_000_000, &random_rows(1, dim, 4))?;
    idx.delete(ids[0])?;
    idx.update(ids[1], &random_rows(1, dim, 5))?;

    let dir = std::env::temp_dir().join("vindex-basic-example");
    idx.save(&dir)?;
    let on_ssd = Index::load(&dir, Numbers::Ssd)?; // full numbers stay in the file: searches only
    let again = on_ssd.search(&questions, 10, Effort::Target(0.95))?;
    let fresh = idx.search(&questions, 10, Effort::Target(0.95))?;
    println!("same answers after loading: {}", again.ids == fresh.ids);
    println!("memory: {} bytes per item", idx.memory().total / idx.memory().items);
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
