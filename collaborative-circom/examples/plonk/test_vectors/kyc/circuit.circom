pragma circom 2.0.0;

include "lib/comparators.circom";

template Kyc(t) {
  signal input blacklist[t];
  signal input country;
  signal input min_age;
  signal input age;

  component blacklist_checks[t];

  for (var i = 0;i<t;i++) {
    blacklist_checks[i] = IsZero();
    blacklist_checks[i].in <== country - blacklist[i];
    blacklist_checks[i].out === 0;
  }

  component geq = GreaterEqThan(8);
  geq.in[0] <== age; 
  geq.in[1] <== min_age; 
  geq.out === 1;
}

component main {public [blacklist, min_age]} = Kyc(3);
