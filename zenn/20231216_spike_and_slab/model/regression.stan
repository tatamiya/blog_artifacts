// The input data is a vector 'y' of length 'N'.
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
  //// expected values
  vector[D] beta0;
  for (i in 1:D) {
    beta0[i] = 0;
  }
  //// variance-covariance matrix
  matrix[D, D] des_mat = X' * X;
  matrix[D, D] beta_cov = inverse(0.5 / N * (des_mat + diag_matrix(diagonal(des_mat))));
  
  // parameters for tau prior 
  real nu = 0.01;
  real r2 = 0.5;
  real sy = variance(Y);
  real s = nu * (1-r2) * sy;
  
  // model definition
  //// priors
  sigma2 ~ inv_gamma(nu/2, s/2);
  beta ~ multi_normal(beta0, sigma2 * beta_cov);
  
  //// target var
  Y ~ normal(mu, sqrt(sigma2));
}

generated quantities {
  vector[N_new] Y_new;
  vector[N_new] mu_new;
  
  mu_new = X_new * beta;
  for (n in 1:N_new) {
    Y_new[n] = normal_rng(mu_new[n], sqrt(sigma2));
  }
}
