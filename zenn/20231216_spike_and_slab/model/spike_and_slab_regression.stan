functions {
  matrix generate_beta_cov_matrix(matrix X){
    int N = dims(X)[1];
    int D = dims(X)[2];
    matrix[D, D] des_mat = X' * X;
    return  inverse(0.5 / N * (des_mat + diag_matrix(diagonal(des_mat))));
  }
  
  real spike_and_slab_regression_lpdf(vector Y, matrix X, vector beta, real sigma2, real nu, real s, array[] int rho, vector rho_pi){
    int D_rho = sum(rho);
    vector[D_rho] beta_rho;
    vector[D_rho] b0_rho;
    int D = size(rho);
    int N = size(Y);
    matrix[N, D_rho] X_rho;
    
    real lp_Y_cond;
    real lp_beta_cond;
    real lp_sigma2;
    
    if (D_rho == 0) {
      // rho_k が全て0の時だけ特別扱い
      lp_Y_cond = normal_lpdf(Y | 0, sqrt(sigma2));
      lp_beta_cond = 0;
      lp_sigma2 = inv_gamma_lpdf(sigma2 | nu/2, s/2);
    } else {
      for (k in 1:D_rho){
        b0_rho[k] = 0;
      }
      int beta_index = 1;
      for (k in 1:D){
        if(rho[k] == 1){
          // rho_k=1 となるk要素だけ抜き出した beta_rho を定義する
          beta_rho[beta_index] = beta[k];
          // 計画行列 X のうち rho_k=1 となる行・列から構成される主部分行列 X_rho を定義する
          X_rho[,beta_index] = col(X, k);
          beta_index += 1;
        }
      }
      matrix[D_rho, D_rho] beta_cov_rho = generate_beta_cov_matrix(X_rho);
      lp_Y_cond = normal_lpdf(Y | X_rho * beta_rho, sqrt(sigma2));
      lp_beta_cond = multi_normal_lpdf(beta_rho | b0_rho, sigma2 * beta_cov_rho);
      lp_sigma2 = inv_gamma_lpdf(sigma2 | nu/2, s/2);
    }
    
    real lp_rho = 0;
    for (i in 1:D) {
      lp_rho += bernoulli_lpmf(rho[i] | rho_pi[i]);
    }
    
    return lp_Y_cond + lp_beta_cond + lp_sigma2 + lp_rho;
  }
}

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

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[D] beta;
  real<lower=0> sigma2;
  
  vector<lower=0, upper=1>[D] rho_pi;
}

transformed parameters {
  vector[N] mu;
  mu = X*beta;
  
  real tau = 1/sigma2;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  vector[to_int(2^D)] lp;
  array[D] int rho;
  for (i in 1:D) {
    rho[i] = 0;
  }
  
  // parameters for sigma^2 prior 
  real nu = 0.01;
  real r2 = 0.5;
  real sy = variance(Y);
  real s = nu * (1-r2) * sy;
  
  // model definition
  // rho=(0,0,...,0) ~ (1,1,...,1) までの全パターンで対数尤度を計算する
  for (k in 1:to_int(2^D)) {
    
    // Y|beta,sigma2,rho の対数尤度を定義 
    lp[k] = spike_and_slab_regression_lpdf(Y | X, beta, sigma2, nu, s, rho, rho_pi);
    
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
    }
  }
  //print("log_sum_exp(lp): ", log_sum_exp(lp));
  target += log_sum_exp(lp);
  
}

// TODO: 事後予測
//generated quantities {
//  vector[N_new] Y_new;
//  vector[N_new] mu_new;
//  array[D] real rho_pi_new;
//  
//  for (i in 1:D) {
//    rho_pi_new[i] = bernoulli_rng(rho_pi[i]);
//  }
//  
//  X_new_rho = ; # X_new から rho_k=1 に対応する列を抜き出す
//  beta_rho = ; # P(beta_rho | rho, Y, sigma2) からサンプリングする
//  
//  mu_new = X_new_rho * beta_rho;
//  for (n in 1:N_new) {
//    Y_new[n] = normal_rng(mu_new[n], sqrt(sigma2));
//  }
//}
