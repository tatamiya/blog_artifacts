data {
  int N;
  int D;
  matrix[N, D] X;
  vector[N] Y;
  
  // for out-of-sample prediction
  int N_new;
  matrix[N_new, D] X_new;
}

// The parameters accepted by the model.
parameters {
  vector[D] beta;
  real<lower=0> sigma2;
}

transformed parameters {
  vector[N] mu;
  mu = X*beta;
  
  real tau = 1/sigma2;
}

// The model to be estimated.
model {
  // parameters for beta prior
  vector[D] beta0;
  array[D] int rho;
  vector[D] rho_v;
  array[D] real rho_pi;
  vector[D] i_D;
  vector[to_int(2^D)] lp;
  for (i in 1:D) {
    beta0[i] = 0;
    rho[i] = 0;
    rho_v[i] = 0;
    rho_pi[i] = 0.5;
    i_D[i] = 1;
  }
  
  //// variance-covariance matrix
  matrix[D, D] des_mat = X' * X;
  matrix[D, D] beta_prec_full = 0.5 / N * (des_mat + diag_matrix(diagonal(des_mat)));
  
  
  // parameters for sigma^2 prior 
  real nu = 0.01;
  real r2 = 0.5;
  real sy = variance(Y);
  real s = nu * (1-r2) * sy;
  
  // model definition
  matrix[D,D] diag_beta_prec = diag_matrix(diagonal(beta_prec_full));
  for (k in 1:to_int(2^D)) {
    matrix[D,D] beta_prec_mat = quad_form_diag(beta_prec_full, rho_v) + N^2 * diag_pre_multiply(i_D - rho_v, diag_beta_prec);
    
    lp[k] = normal_lpdf(Y | X * beta, sqrt(sigma2)) + multi_normal_lpdf(beta | beta0, sigma2 * inverse(beta_prec_mat))  + inv_gamma_lpdf(sigma2 | nu/2, s/2);
    
    // P(rho) の対数尤度を加算する
    for (i in 1:D) {
      lp[k] += bernoulli_lpmf(rho[i] | rho_pi[i]);
    }
    
    // ベクトル rho を更新
    int rho_increment = 1;
    for (i in 1:D) {
      int tmp_added = rho[i] + rho_increment;
      if (tmp_added==2) {
        rho[i]=0;
      } else {
        rho[i] = tmp_added;
        rho_increment = 0;
      }
      rho_v[i] = rho[i];
    }
  }
  //print("log_sum_exp(lp): ", log_sum_exp(lp));
  target += log_sum_exp(lp);
  
}

generated quantities {
  vector[N_new] Y_new;
  vector[N_new] mu_new;
  
  mu_new = X_new * beta;
  for (n in 1:N_new) {
    Y_new[n] = normal_rng(mu_new[n], sqrt(sigma2));
  }
}
