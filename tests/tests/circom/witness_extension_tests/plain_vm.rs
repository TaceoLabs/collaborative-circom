use ark_bn254::Bn254;
use circom_mpc_compiler::CoCircomCompiler;
use circom_mpc_compiler::CompilerConfig;
use circom_mpc_vm::mpc_vm::VMConfig;
use circom_types::Witness;
use co_circom_snarks::SharedWitness;
use std::{
    fs::{self, File},
    str::FromStr,
};

pub struct TestInputs {
    inputs: Vec<Vec<ark_bn254::Fr>>,
    witnesses: Vec<Witness<ark_ff::Fp<ark_ff::MontBackend<ark_bn254::FrConfig, 4>, 4>>>,
}

fn convert_witness(mut witness: SharedWitness<ark_bn254::Fr, ark_bn254::Fr>) -> Vec<ark_bn254::Fr> {
    witness.public_inputs.extend(witness.witness);
    witness.public_inputs
}

fn read_field_element(s: &str) -> ark_bn254::Fr {
    if let Some(striped) = s.strip_prefix('-') {
        -ark_bn254::Fr::from_str(striped).unwrap()
    } else {
        ark_bn254::Fr::from_str(s).unwrap()
    }
}
macro_rules! witness_extension_test_plain {
    ($name: ident) => {
        #[test]
        fn $name() {
            let inp: TestInputs = from_test_name(stringify!($name));
            for i in 0..inp.inputs.len() {
                let mut compiler_config = CompilerConfig::default();
                compiler_config
                    .link_library
                    .push("../test_vectors/WitnessExtension/tests/libs/".into());
                let parsed = CoCircomCompiler::<Bn254>::parse(
                    format!(
                        "../test_vectors/WitnessExtension/tests/{}.circom",
                        stringify!($name)
                    ),
                    compiler_config,
                )
                .unwrap();
                let is_witness = parsed
                    .to_plain_vm(VMConfig::default())
                    .run_with_flat(inp.inputs[i].to_owned(), 0)
                    .unwrap()
                    .into_shared_witness();
                assert_eq!(convert_witness(is_witness), inp.witnesses[i].values);
            }
        }
    };

    ($name: ident, $file: expr, $input: expr, $should:expr) => {
        witness_extension_test_plain!($name, $file, $input, $should, "witness");
    };

    ($name: ident, $file: expr, $input: expr) => {
        witness_extension_test_plain!($name, $file, $input, $file);
    };
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
    println!("i: {}", i);
    TestInputs { inputs, witnesses }
}

witness_extension_test_plain!(aliascheck_test);
witness_extension_test_plain!(babyadd_tester);
witness_extension_test_plain!(babycheck_test);
witness_extension_test_plain!(babypbk_test);
witness_extension_test_plain!(binsub_test);
witness_extension_test_plain!(binsum_test);
witness_extension_test_plain!(constants_test);
witness_extension_test_plain!(control_flow);
witness_extension_test_plain!(eddsa_test);
witness_extension_test_plain!(eddsa_verify);
witness_extension_test_plain!(eddsamimc_test);
witness_extension_test_plain!(eddsaposeidon_test);
witness_extension_test_plain!(edwards2montgomery);
witness_extension_test_plain!(escalarmul_test);
witness_extension_test_plain!(escalarmul_test_min);
witness_extension_test_plain!(escalarmulany_test);
witness_extension_test_plain!(escalarmulfix_test);
witness_extension_test_plain!(escalarmulw4table);
witness_extension_test_plain!(escalarmulw4table_test);
witness_extension_test_plain!(escalarmulw4table_test3);
witness_extension_test_plain!(functions);
witness_extension_test_plain!(greatereqthan);
witness_extension_test_plain!(greaterthan);
witness_extension_test_plain!(isequal);
witness_extension_test_plain!(iszero);
witness_extension_test_plain!(lesseqthan);
witness_extension_test_plain!(lessthan);
witness_extension_test_plain!(mimc_hasher);
witness_extension_test_plain!(mimc_sponge_hash_test);
witness_extension_test_plain!(mimc_sponge_test);
witness_extension_test_plain!(mimc_test);
witness_extension_test_plain!(montgomery2edwards);
witness_extension_test_plain!(montgomeryadd);
witness_extension_test_plain!(montgomerydouble);
witness_extension_test_plain!(multiplier16);
witness_extension_test_plain!(multiplier2);
witness_extension_test_plain!(mux1_1);
witness_extension_test_plain!(mux2_1);
witness_extension_test_plain!(mux3_1);
witness_extension_test_plain!(mux4_1);
witness_extension_test_plain!(pedersen2_test);
witness_extension_test_plain!(pedersen_hasher);
witness_extension_test_plain!(pedersen_test);
witness_extension_test_plain!(pointbits_loopback);
witness_extension_test_plain!(poseidon3_test);
witness_extension_test_plain!(poseidon6_test);
witness_extension_test_plain!(poseidon_hasher1);
witness_extension_test_plain!(poseidon_hasher16);
witness_extension_test_plain!(poseidon_hasher2);
witness_extension_test_plain!(poseidonex_test);
witness_extension_test_plain!(sha256_2_test);
witness_extension_test_plain!(sha256_test448);
witness_extension_test_plain!(sha256_test512);
witness_extension_test_plain!(shared_control_flow);
witness_extension_test_plain!(shared_control_flow_arrays);
witness_extension_test_plain!(sign_test);
witness_extension_test_plain!(sqrt_test);
witness_extension_test_plain!(smtprocessor10_test);
witness_extension_test_plain!(smtverifier10_test);
witness_extension_test_plain!(sum_test);
witness_extension_test_plain!(winner);
witness_extension_test_plain!(bitonic_sort);
