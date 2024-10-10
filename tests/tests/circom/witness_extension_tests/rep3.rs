use ark_bn254::Bn254;
use circom_mpc_compiler::CoCircomCompiler;
use circom_types::Witness;
use co_circom_snarks::SharedWitness;
use itertools::izip;
use mpc_core::protocols::rep3::{self};
use rand::thread_rng;
use std::fs;
use std::str::FromStr;
use std::{fs::File, thread};
use tests::rep3_network::Rep3TestNetwork;

use circom_mpc_compiler::CompilerConfig;
use circom_mpc_vm::{mpc_vm::VMConfig, Rep3VmType};

#[allow(dead_code)]
fn install_tracing() {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

pub struct TestInputs {
    inputs: Vec<Vec<ark_bn254::Fr>>,
    witnesses: Vec<Witness<ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>>>,
}

fn combine_field_elements_for_vm(
    a: SharedWitness<ark_bn254::Fr, rep3::arithmetic::FieldShare<ark_bn254::Fr>>,
    b: SharedWitness<ark_bn254::Fr, rep3::arithmetic::FieldShare<ark_bn254::Fr>>,
    c: SharedWitness<ark_bn254::Fr, rep3::arithmetic::FieldShare<ark_bn254::Fr>>,
) -> Vec<ark_bn254::Fr> {
    let mut res = Vec::with_capacity(a.public_inputs.len() + a.witness.len());
    for (a, b, c) in izip!(a.public_inputs, b.public_inputs, c.public_inputs) {
        assert_eq!(a, b);
        assert_eq!(b, c);
        res.push(a);
    }
    res.extend(rep3::combine_field_elements(
        a.witness, b.witness, c.witness,
    ));
    res
}

fn read_field_element(s: &str) -> ark_bn254::Fr {
    if let Some(striped) = s.strip_prefix('-') {
        -ark_bn254::Fr::from_str(striped).unwrap()
    } else {
        ark_bn254::Fr::from_str(s).unwrap()
    }
}
pub fn from_test_name(fn_name: &str) -> TestInputs {
    let mut witnesses: Vec<Witness<ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>>> =
        Vec::new();
    let mut inputs: Vec<Vec<ark_bn254::Fr>> = Vec::new();
    let mut i = 0;
    loop {
        if fs::metadata(format!(
            "../test_vectors/WitnessExtension/kats/{}/witness{}.wtns",
            fn_name, i
        ))
        .is_err()
        {
            break;
        }
        let witness = File::open(format!(
            "../test_vectors/WitnessExtension/kats/{}/witness{}.wtns",
            fn_name, i
        ))
        .unwrap();
        let should_witness = Witness::<ark_bn254::Fr>::from_reader(witness).unwrap();
        witnesses.push(should_witness);
        let input_file = File::open(format!(
            "../test_vectors/WitnessExtension/kats/{}/input{}.json",
            fn_name, i
        ))
        .unwrap();
        let json_str: serde_json::Value = serde_json::from_reader(input_file).unwrap();
        let input = json_str
            .get("in")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|s| read_field_element(s.as_str().unwrap()))
            .collect::<Vec<_>>();
        inputs.push(input);
        i += 1
    }
    TestInputs { inputs, witnesses }
}

macro_rules! run_test {
    ($file: expr, $input: expr) => {{
        //install_tracing();
        let mut rng = thread_rng();
        let inputs = rep3::share_field_elements($input, &mut rng);
        let test_network = Rep3TestNetwork::default();
        let mut threads = vec![];

        for (net, input) in izip!(test_network.get_party_networks(), inputs) {
            threads.push(thread::spawn(move || {
                let mut compiler_config = CompilerConfig::default();
                compiler_config
                    .link_library
                    .push("../test_vectors/WitnessExtension/tests/libs/".into());
                let mut witness_extension =
                    CoCircomCompiler::<Bn254>::parse($file.to_owned(), compiler_config)
                        .unwrap()
                        .to_rep3_vm_with_network(net, VMConfig::default())
                        .unwrap();
                witness_extension
                    .run_with_flat(
                        input
                            .into_iter()
                            .map(|x| Rep3VmType::Arithmetic(x))
                            .collect(),
                        0,
                    )
                    .unwrap()
                    .into_shared_witness()
            }));
        }
        let result3 = threads.pop().unwrap().join().unwrap();
        let result2 = threads.pop().unwrap().join().unwrap();
        let result1 = threads.pop().unwrap().join().unwrap();
        combine_field_elements_for_vm(result1, result2, result3)
    }};
}

macro_rules! witness_extension_test_rep3 {
    ($name: ident) => {
        #[test]
        fn $name() {
            let inp: TestInputs = from_test_name(stringify!($name));
            // let path = inp.circuit_path.as_str().to_owned();
            for i in 0..inp.inputs.len() {
                let is_witness = run_test!(
                    format!(
                        "../test_vectors/WitnessExtension/tests/{}.circom",
                        stringify!($name)
                    ),
                    &inp.inputs[i]
                );
                assert_eq!(is_witness, inp.witnesses[i].values);
            }
        }
    };

    ($name: ident, $file: expr, $input: expr, $should:expr) => {
        witness_extension_test!($name, $file, $input, $should, "witness");
    };

    ($name: ident, $file: expr, $input: expr) => {
        witness_extension_test!($name, $file, $input, $file);
    };
}
macro_rules! witness_extension_test_rep3_ignored {
    ($name: ident) => {
        #[test]
        #[ignore]
        fn $name() {
            let inp: TestInputs = from_test_name(stringify!($name));
            // let path = inp.circuit_path.as_str().to_owned();
            for i in 0..inp.inputs.len() {
                let is_witness = run_test!(
                    format!(
                        "../test_vectors/WitnessExtension/tests/{}.circom",
                        stringify!($name)
                    ),
                    &inp.inputs[i]
                );
                assert_eq!(is_witness, inp.witnesses[i].values);
            }
        }
    };

    ($name: ident, $file: expr, $input: expr, $should:expr) => {
        witness_extension_test!($name, $file, $input, $should, "witness");
    };

    ($name: ident, $file: expr, $input: expr) => {
        witness_extension_test!($name, $file, $input, $file);
    };
}

witness_extension_test_rep3!(aliascheck_test);
witness_extension_test_rep3!(babyadd_tester);
witness_extension_test_rep3!(babycheck_test);
witness_extension_test_rep3!(babypbk_test);
witness_extension_test_rep3!(binsub_test);
witness_extension_test_rep3!(binsum_test);
witness_extension_test_rep3!(constants_test);
witness_extension_test_rep3!(control_flow);
witness_extension_test_rep3_ignored!(eddsa_test);
witness_extension_test_rep3_ignored!(eddsa_verify);
witness_extension_test_rep3_ignored!(eddsamimc_test);
witness_extension_test_rep3_ignored!(eddsaposeidon_test);
witness_extension_test_rep3!(edwards2montgomery);
witness_extension_test_rep3!(escalarmul_test);
witness_extension_test_rep3!(escalarmul_test_min);
witness_extension_test_rep3!(escalarmulany_test);
witness_extension_test_rep3_ignored!(escalarmulfix_test);
witness_extension_test_rep3!(escalarmulw4table);
witness_extension_test_rep3!(escalarmulw4table_test);
witness_extension_test_rep3!(escalarmulw4table_test3);
witness_extension_test_rep3!(functions);
witness_extension_test_rep3!(greatereqthan);
witness_extension_test_rep3!(greaterthan);
witness_extension_test_rep3!(isequal);
witness_extension_test_rep3!(iszero);
witness_extension_test_rep3!(lesseqthan);
witness_extension_test_rep3!(lessthan);
witness_extension_test_rep3!(mimc_hasher);
witness_extension_test_rep3!(mimc_sponge_hash_test);
witness_extension_test_rep3!(mimc_sponge_test);
witness_extension_test_rep3!(mimc_test);
witness_extension_test_rep3!(montgomery2edwards);
witness_extension_test_rep3!(montgomeryadd);
witness_extension_test_rep3!(montgomerydouble);
witness_extension_test_rep3!(multiplier16);
witness_extension_test_rep3!(multiplier2);
witness_extension_test_rep3!(mux1_1);
witness_extension_test_rep3!(mux2_1);
witness_extension_test_rep3!(mux3_1);
witness_extension_test_rep3!(mux4_1);
witness_extension_test_rep3_ignored!(pedersen2_test);
witness_extension_test_rep3!(pedersen_hasher);
witness_extension_test_rep3_ignored!(pedersen_test);
witness_extension_test_rep3!(pointbits_loopback);
witness_extension_test_rep3!(poseidon3_test);
witness_extension_test_rep3!(poseidon6_test);
witness_extension_test_rep3!(poseidon_hasher1);
witness_extension_test_rep3!(poseidon_hasher16);
witness_extension_test_rep3!(poseidon_hasher2);
witness_extension_test_rep3!(poseidonex_test);
witness_extension_test_rep3_ignored!(sha256_2_test);
witness_extension_test_rep3_ignored!(sha256_test448);
witness_extension_test_rep3_ignored!(sha256_test512);
witness_extension_test_rep3!(shared_control_flow);
witness_extension_test_rep3!(shared_control_flow_arrays);
witness_extension_test_rep3!(sign_test);
witness_extension_test_rep3!(sqrt_test);
witness_extension_test_rep3!(smtprocessor10_test);
witness_extension_test_rep3!(smtverifier10_test);
witness_extension_test_rep3!(sum_test);
witness_extension_test_rep3!(winner);
witness_extension_test_rep3!(bitonic_sort);
