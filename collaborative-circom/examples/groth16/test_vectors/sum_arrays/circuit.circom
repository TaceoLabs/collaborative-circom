pragma circom 2.0.0;

template Main(t) {
  signal input a[t];
  signal input b[t];
  signal input c[t];
  signal ab[t];
  signal abc[t];
  signal acc[t];
  signal out;
  assert(t>=2);
  for(var i = 0;i<t;i++) {
    ab[i] <== a[i] + b[i];
    abc[i] <== ab[i] + c[i];
  }
  acc[0] <== abc[0];
  for (var i = 1;i<t;i++) {
    acc[i] <== acc[i-1] + abc[i];
  }
  out <== acc[t-1];
  log("result", out);
}

component main{public [b, c]} = Main(3);
