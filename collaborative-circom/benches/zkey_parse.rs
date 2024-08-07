use ark_bn254::Bn254;
use criterion::{criterion_group, criterion_main, Criterion};
use std::fs::File;

fn groth16_zkey_parse(c: &mut Criterion) {
    c.bench_function("groth16 zkey parse", |b| {
        b.iter(|| {
            let zkey = File::open("../test_vectors/bn254/multiplier2/multiplier2.zkey").unwrap();
            circom_types::groth16::zkey::ZKey::<Bn254>::from_reader(zkey).unwrap();
        })
    });
}

fn plonk_zkey_parse(c: &mut Criterion) {
    c.bench_function("plonk zkey parse", |b| {
        b.iter(|| {
            let zkey = File::open("../test_vectors/Plonk/bn254/multiplier2.zkey").unwrap();
            circom_types::plonk::ZKey::<Bn254>::from_reader(zkey).unwrap();
        })
    });
}

criterion_group!(benches, groth16_zkey_parse, plonk_zkey_parse);
criterion_main!(benches);
