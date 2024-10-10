mod field_share {
    use ark_ff::Field;
    use ark_std::{UniformRand, Zero};
    use itertools::{izip, Itertools};
    use mpc_core::protocols::shamir::{self, arithmetic, ShamirPreprocessing};
    use rand::thread_rng;
    use std::{str::FromStr, thread};
    use tests::shamir_network::ShamirTestNetwork;
    use tokio::sync::oneshot;

    async fn shamir_add_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let x = ark_bn254::Fr::rand(&mut rng);
        let y = ark_bn254::Fr::rand(&mut rng);
        let x_shares = shamir::share_field_element(x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_field_element(y, threshold, num_parties, &mut rng);
        let should_result = x + y;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, x, y) in izip!(tx, x_shares, y_shares) {
            thread::spawn(move || tx.send(arithmetic::add(x, y)));
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_element(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_add() {
        shamir_add_inner(3, 1).await;
        shamir_add_inner(10, 4).await;
    }

    async fn shamir_sub_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let x = ark_bn254::Fr::rand(&mut rng);
        let y = ark_bn254::Fr::rand(&mut rng);
        let x_shares = shamir::share_field_element(x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_field_element(y, threshold, num_parties, &mut rng);
        let should_result = x - y;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, x, y) in izip!(tx, x_shares, y_shares) {
            thread::spawn(move || tx.send(arithmetic::sub(x, y)));
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_element(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_sub() {
        shamir_sub_inner(3, 1).await;
        shamir_sub_inner(10, 4).await;
    }

    async fn shamir_mul2_then_add_inner(num_parties: usize, threshold: usize) {
        let test_network = ShamirTestNetwork::new(num_parties);
        let mut rng = thread_rng();
        let x = ark_bn254::Fr::rand(&mut rng);
        let y = ark_bn254::Fr::rand(&mut rng);
        let x_shares = shamir::share_field_element(x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_field_element(y, threshold, num_parties, &mut rng);
        let should_result = ((x * y) * y) + x;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (net, tx, x, y) in izip!(test_network.get_party_networks(), tx, x_shares, y_shares) {
            tokio::spawn(async move {
                let mut shamir = ShamirPreprocessing::new(threshold, net, 2)
                    .await
                    .unwrap()
                    .into();
                let mul = arithmetic::mul(x, y, &mut shamir).await.unwrap();
                let mul = arithmetic::mul(mul, y, &mut shamir).await.unwrap();
                tx.send(arithmetic::add(mul, x))
            });
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_element(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_mul2_then_add() {
        shamir_mul2_then_add_inner(3, 1).await;
        shamir_mul2_then_add_inner(10, 4).await;
    }

    async fn shamir_mul_vec_bn_inner(num_parties: usize, threshold: usize) {
        let test_network = ShamirTestNetwork::new(num_parties);
        let mut rng = thread_rng();
        let x = [
            ark_bn254::Fr::from_str(
                "13839525561076761625780930844889299788193703994911163378019280196128582690055",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "19302971480864839163158232064620707211435225928426123775531639309944891593977",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "8048717310762513532550620831072439583505607813129662608591015555880153427210",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "2585271390974436123003027749932103593962191064365118925254473311197989280023",
            )
            .unwrap(),
        ];
        let y = [
            ark_bn254::Fr::from_str(
                "2688648969035332064113669477511029957484512453056743431884706385750388613065",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "13632770404954969699480437686769008635735921498648460325387842712839596176806",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "19199593902803943133889170931116903997086625101975591190159463567024116566625",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "8255472466884305547009533395117607586789669747151273739964395707537515634749",
            )
            .unwrap(),
        ];
        let should_result = vec![
            ark_bn254::Fr::from_str(
                "14012338922664984944451142760937475581748095944353358534203030914664561190462",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "4297594441150501195973997511775989720904927516253689527653694984160382713321",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "7875903949174289914141782934879682497141865775307179984684659764891697566272",
            )
            .unwrap(),
            ark_bn254::Fr::from_str(
                "6646526994769136778802685410292764833027657364709823469005920616147071273574",
            )
            .unwrap(),
        ];

        let x_shares = shamir::share_field_elements(&x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_field_elements(&y, threshold, num_parties, &mut rng);

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (net, tx, x, y) in izip!(test_network.get_party_networks(), tx, x_shares, y_shares) {
            tokio::spawn(async move {
                let mut shamir = ShamirPreprocessing::new(threshold, net, x.len())
                    .await
                    .unwrap()
                    .into();
                let mul = arithmetic::mul_vec(&x, &y, &mut shamir).await.unwrap();
                tx.send(mul)
            });
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_elements(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_mul_vec_bn() {
        shamir_mul_vec_bn_inner(3, 1).await;
        shamir_mul_vec_bn_inner(10, 4).await;
    }

    async fn shamir_mul_vec_inner(num_parties: usize, threshold: usize) {
        let test_network = ShamirTestNetwork::new(num_parties);
        let mut rng = thread_rng();
        let x = (0..1)
            .map(|_| ark_bn254::Fr::from_str("2").unwrap())
            .collect::<Vec<_>>();
        let y = (0..x.len())
            .map(|_| ark_bn254::Fr::from_str("3").unwrap())
            .collect::<Vec<_>>();

        let mut should_result = Vec::with_capacity(x.len());
        for (x, y) in x.iter().zip(y.iter()) {
            should_result.push((x * y) * y);
        }

        let x_shares = shamir::share_field_elements(&x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_field_elements(&y, threshold, num_parties, &mut rng);

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (net, tx, x, y) in izip!(test_network.get_party_networks(), tx, x_shares, y_shares) {
            tokio::spawn(async move {
                let mut shamir = ShamirPreprocessing::new(threshold, net, x.len() * 2)
                    .await
                    .unwrap()
                    .into();
                let mul = arithmetic::mul_vec(&x, &y, &mut shamir).await.unwrap();
                let mul = arithmetic::mul_vec(&mul, &y, &mut shamir).await.unwrap();
                tx.send(mul)
            });
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_elements(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_mul_vec() {
        shamir_mul_vec_inner(3, 1).await;
        shamir_mul_vec_inner(10, 4).await;
    }

    async fn shamir_neg_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let x = ark_bn254::Fr::rand(&mut rng);
        let x_shares = shamir::share_field_element(x, threshold, num_parties, &mut rng);
        let should_result = -x;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, x) in izip!(tx, x_shares) {
            thread::spawn(move || tx.send(arithmetic::neg(x)));
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_element(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_neg() {
        shamir_neg_inner(3, 1).await;
        shamir_neg_inner(10, 4).await;
    }

    async fn shamir_inv_inner(num_parties: usize, threshold: usize) {
        let test_network = ShamirTestNetwork::new(num_parties);
        let mut rng = thread_rng();
        let mut x = ark_bn254::Fr::rand(&mut rng);
        while x.is_zero() {
            x = ark_bn254::Fr::rand(&mut rng);
        }
        let x_shares = shamir::share_field_element(x, threshold, num_parties, &mut rng);
        let should_result = x.inverse().unwrap();

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (net, tx, x) in izip!(test_network.get_party_networks(), tx, x_shares) {
            tokio::spawn(async move {
                let mut shamir = ShamirPreprocessing::new(threshold, net, 1)
                    .await
                    .unwrap()
                    .into();
                tx.send(arithmetic::inv(x, &mut shamir).await.unwrap())
            });
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_field_element(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_inv() {
        shamir_inv_inner(3, 1).await;
        shamir_inv_inner(10, 4).await;
    }
}

mod curve_share {
    use std::thread;

    use ark_ff::UniformRand;
    use itertools::{izip, Itertools};
    use mpc_core::protocols::shamir::{self, pointshare};
    use rand::thread_rng;
    use tokio::sync::oneshot;

    async fn shamir_add_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let x = ark_bn254::G1Projective::rand(&mut rng);
        let y = ark_bn254::G1Projective::rand(&mut rng);
        let x_shares = shamir::share_curve_point(x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_curve_point(y, threshold, num_parties, &mut rng);
        let should_result = x + y;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, x, y) in izip!(tx, x_shares, y_shares) {
            thread::spawn(move || tx.send(pointshare::add(&x, &y)));
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_curve_point(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_add() {
        shamir_add_inner(3, 1).await;
        shamir_add_inner(10, 4).await;
    }

    async fn shamir_sub_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let x = ark_bn254::G1Projective::rand(&mut rng);
        let y = ark_bn254::G1Projective::rand(&mut rng);
        let x_shares = shamir::share_curve_point(x, threshold, num_parties, &mut rng);
        let y_shares = shamir::share_curve_point(y, threshold, num_parties, &mut rng);
        let should_result = x - y;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, x, y) in izip!(tx, x_shares, y_shares) {
            thread::spawn(move || tx.send(pointshare::sub(&x, &y)));
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_curve_point(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_sub() {
        shamir_sub_inner(3, 1).await;
        shamir_sub_inner(10, 4).await;
    }

    async fn shamir_scalar_mul_public_point_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let public_point = ark_bn254::G1Projective::rand(&mut rng);
        let scalar = ark_bn254::Fr::rand(&mut rng);
        let scalar_shares = shamir::share_field_element(scalar, threshold, num_parties, &mut rng);
        let should_result = public_point * scalar;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, scalar) in izip!(tx, scalar_shares) {
            thread::spawn(move || {
                tx.send(pointshare::scalar_mul_public_point(scalar, &public_point))
            });
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_curve_point(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_scalar_mul_public_point() {
        shamir_scalar_mul_public_point_inner(3, 1).await;
        shamir_scalar_mul_public_point_inner(10, 4).await;
    }

    async fn shamir_scalar_mul_public_scalar_inner(num_parties: usize, threshold: usize) {
        let mut rng = thread_rng();
        let point = ark_bn254::G1Projective::rand(&mut rng);
        let public_scalar = ark_bn254::Fr::rand(&mut rng);
        let point_shares = shamir::share_curve_point(point, threshold, num_parties, &mut rng);
        let should_result = point * public_scalar;

        let mut tx = Vec::with_capacity(num_parties);
        let mut rx = Vec::with_capacity(num_parties);
        for _ in 0..num_parties {
            let (t, r) = oneshot::channel();
            tx.push(t);
            rx.push(r);
        }

        for (tx, point) in izip!(tx, point_shares) {
            thread::spawn(move || {
                tx.send(pointshare::scalar_mul_public_scalar(&point, &public_scalar))
            });
        }

        let mut results = Vec::with_capacity(num_parties);
        for r in rx {
            results.push(r.await.unwrap());
        }

        let is_result =
            shamir::combine_curve_point(&results, &(1..=num_parties).collect_vec(), threshold)
                .unwrap();

        assert_eq!(is_result, should_result);
    }

    #[tokio::test]
    async fn shamir_scalar_mul_public_scalar() {
        shamir_scalar_mul_public_scalar_inner(3, 1).await;
        shamir_scalar_mul_public_scalar_inner(10, 4).await;
    }
}
