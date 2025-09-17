clear; close all; clc;

%% Binomial
% Coin flip is binomial

p_true = 0.7; % True probability we are estimating
m = 5; % Each observation consists of 5 coin flips
n_max = 100; % We get 100 observations
n_mc = 1000; % Monte Carlo trials (run 1000 simulations)


alpha1 = 1; beta1 = 1; % Uniform assume not loaded die (mean=0.5)
alpha2 = 7; beta2 = 3; % Good estimate (mean=0.7)
alpha3 = 2; beta3 = 8; % Bad estimate (mean=0.2)

MSE_ML_binom = zeros(n_max, 1);
MSE_prior1_binom = zeros(n_max, 1);
MSE_prior2_binom = zeros(n_max, 1);
MSE_prior3_binom = zeros(n_max, 1);

for trial = 1:n_mc % for each trial
    data = binornd(m, p_true, n_max, 1); % generate 100 observations each with 5 coin flips
    
    % Save first dataset for posterior plot
    if trial == 1
        data_saved_binom = data;
    end
    for n = 1:n_max
        successes = sum(data(1:n)); % total successes so far
        trials = n * m; % total coin flips so far
        
        % ML estimate which is just successes/trials
        p_ML = successes / trials;
        % get the bayesian estimate for each prior which is adding on the
        % prior successes (alpha) and the prior failures (beta)
        p_Bayes1 = (alpha1 + successes) / (alpha1 + beta1 + trials);
        p_Bayes2 = (alpha2 + successes) / (alpha2 + beta2 + trials);
        p_Bayes3 = (alpha3 + successes) / (alpha3 + beta3 + trials);
        
        % error is just how wrong each estimate is
        MSE_ML_binom(n) = MSE_ML_binom(n) + (p_ML - p_true)^2;
        MSE_prior1_binom(n) = MSE_prior1_binom(n) + (p_Bayes1 - p_true)^2;
        MSE_prior2_binom(n) = MSE_prior2_binom(n) + (p_Bayes2 - p_true)^2;
        MSE_prior3_binom(n) = MSE_prior3_binom(n) + (p_Bayes3 - p_true)^2;
    end
end
% Average MSE which is just divided by the number of monte carlo
% simulations
MSE_ML_binom = MSE_ML_binom / n_mc;
MSE_prior1_binom = MSE_prior1_binom / n_mc;
MSE_prior2_binom = MSE_prior2_binom / n_mc;
MSE_prior3_binom = MSE_prior3_binom / n_mc;

%% Gaussian with Known Variance 

mu_true = 5; % True mean we are estimating
sigma_known = 2; % Known standard deviation


mu0_1 = 0; sigma0_1 = 10;  % Vague prior (far from truth, high uncertainty)
mu0_2 = 5; sigma0_2 = 1;   % Good estimate (mean=5)
mu0_3 = 10; sigma0_3 = 0.5; % Bad estimate (mean=10, high confidence)

MSE_ML_gauss1 = zeros(n_max, 1);
MSE_prior1_gauss1 = zeros(n_max, 1);
MSE_prior2_gauss1 = zeros(n_max, 1);
MSE_prior3_gauss1 = zeros(n_max, 1);

for trial = 1:n_mc % for each trial
    data = normrnd(mu_true, sigma_known, n_max, 1); % generate 100 observations from normal distribution

    if trial == 1
        data_saved_gauss1 = data;
    end
    for n = 1:n_max
        
        % ML estimate which is just the sample mean
        mu_ML = mean(data(1:n));
        
        % bayesian estimate for each prior using precision weighting
        % precision = 1/variance, and precisions add
        tau_data = n/sigma_known^2; % total data precision
        

        tau0_1 = 1/sigma0_1^2; % prior precision
        mu_Bayes1 = (tau0_1*mu0_1 + tau_data*mean(data(1:n))) / (tau0_1 + tau_data);
        
        tau0_2 = 1/sigma0_2^2;
        mu_Bayes2 = (tau0_2*mu0_2 + tau_data*mean(data(1:n))) / (tau0_2 + tau_data);
        
        tau0_3 = 1/sigma0_3^2;
        mu_Bayes3 = (tau0_3*mu0_3 + tau_data*mean(data(1:n))) / (tau0_3 + tau_data);
        
        % error is just how wrong each estimate is
        MSE_ML_gauss1(n) = MSE_ML_gauss1(n) + (mu_ML - mu_true)^2;
        MSE_prior1_gauss1(n) = MSE_prior1_gauss1(n) + (mu_Bayes1 - mu_true)^2;
        MSE_prior2_gauss1(n) = MSE_prior2_gauss1(n) + (mu_Bayes2 - mu_true)^2;
        MSE_prior3_gauss1(n) = MSE_prior3_gauss1(n) + (mu_Bayes3 - mu_true)^2;
    end
end

% Average MSE which is just divided by the number of monte carlo
% simulations
MSE_ML_gauss1 = MSE_ML_gauss1 / n_mc;
MSE_prior1_gauss1 = MSE_prior1_gauss1 / n_mc;
MSE_prior2_gauss1 = MSE_prior2_gauss1 / n_mc;
MSE_prior3_gauss1 = MSE_prior3_gauss1 / n_mc;

%% Gaussian with Known Mean 

mu_known = 0; % Known mean (often centered at 0)
sigma2_true = 4; % True variance we are estimating (variance=4 means std dev=2)

% mean = beta/(alpha-1) for alpha > 1
alpha1_ig = 2; beta1_ig = 2;   % Weak prior (mean = 2)
alpha2_ig = 5; beta2_ig = 16;  % Good estimate (mean = 4)
alpha3_ig = 10; beta3_ig = 10; % Bad estimate (mean = 1.11)

MSE_ML_gauss2 = zeros(n_max, 1);
MSE_prior1_gauss2 = zeros(n_max, 1);
MSE_prior2_gauss2 = zeros(n_max, 1);
MSE_prior3_gauss2 = zeros(n_max, 1);

for trial = 1:n_mc % for each trial
    data = normrnd(mu_known, sqrt(sigma2_true), n_max, 1); % generate 100 observations (took sqrt because stddev)
    if trial == 1
        data_saved_gauss2 = data;
    end
    for n = 1:n_max
        % Sum of squared deviations from known mean
        S = sum((data(1:n) - mu_known).^2);
        % ML estimate which is just S/n
        sigma2_ML = S / n;
        
        % get the bayesian estimate for each prior
        % Posterior is IG(alpha + n/2, beta + S/2)

        alpha_post1 = alpha1_ig + n/2; % add half the observations to shape
        beta_post1 = beta1_ig + S/2; % add half the sum of squares to scale
        sigma2_Bayes1 = beta_post1 / (alpha_post1 - 1); % posterior mean
        
        alpha_post2 = alpha2_ig + n/2;
        beta_post2 = beta2_ig + S/2;
        sigma2_Bayes2 = beta_post2 / (alpha_post2 - 1);
        
        alpha_post3 = alpha3_ig + n/2;
        beta_post3 = beta3_ig + S/2;
        sigma2_Bayes3 = beta_post3 / (alpha_post3 - 1);
        
        % error is just how wrong each estimate is
        MSE_ML_gauss2(n) = MSE_ML_gauss2(n) + (sigma2_ML - sigma2_true)^2;
        MSE_prior1_gauss2(n) = MSE_prior1_gauss2(n) + (sigma2_Bayes1 - sigma2_true)^2;
        MSE_prior2_gauss2(n) = MSE_prior2_gauss2(n) + (sigma2_Bayes2 - sigma2_true)^2;
        MSE_prior3_gauss2(n) = MSE_prior3_gauss2(n) + (sigma2_Bayes3 - sigma2_true)^2;
    end
end
% Average MSE which is just divided by the number of monte carlo
% simulations
MSE_ML_gauss2 = MSE_ML_gauss2 / n_mc;
MSE_prior1_gauss2 = MSE_prior1_gauss2 / n_mc;
MSE_prior2_gauss2 = MSE_prior2_gauss2 / n_mc;
MSE_prior3_gauss2 = MSE_prior3_gauss2 / n_mc;

%% All Methods on One Plot
figure('Position', [100, 100, 1800, 900]);

% Binomial plots
subplot(3,2,1);
% Plot MSE Comparison - basically MSE on the y axis and the number of
% observations on the x axis
plot(1:n_max, MSE_ML_binom, 'k--', 'LineWidth', 2); hold on;
plot(1:n_max, MSE_prior1_binom, 'b-', 'LineWidth', 1.5);
plot(1:n_max, MSE_prior2_binom, 'r-', 'LineWidth', 1.5);
plot(1:n_max, MSE_prior3_binom, 'g-', 'LineWidth', 1.5);
xlabel('Observations'); ylabel('MSE');
title('Binomial: Estimating p');
legend('ML', 'Uniform', 'Good Prior', 'Bad Prior', 'Location', 'best');
grid on;

subplot(3,2,2);
% Plot Posterior Evolution - basically the posterior density on the y axis
% and the actual probability on the x axis
n_points = [0, 5, 10, 25, 50, 100]; % for example n = 0 is just beta(7,3)
p_range = 0:0.001:1;
for i = 1:length(n_points)
    n = n_points(i);
    if n == 0
        % Just the prior
        alpha_post = alpha2; beta_post = beta2;
    else
        % Posterior - using saved data
        s = sum(data_saved_binom(1:n));
        alpha_post = alpha2 + s;
        beta_post = beta2 + n*m - s;
    end
    plot(p_range, betapdf(p_range, alpha_post, beta_post), 'LineWidth', 1.5); hold on;
end
xline(p_true, 'k--', 'LineWidth', 2);
xlabel('p'); ylabel('Density');
title('Beta Posterior Evolution');
grid on;

% Gaussian Mean plots
subplot(3,2,3);
% Plot MSE Comparison - basically MSE on the y axis and the number of
% observations on the x axis
plot(1:n_max, MSE_ML_gauss1, 'k--', 'LineWidth', 2); hold on;
plot(1:n_max, MSE_prior1_gauss1, 'b-', 'LineWidth', 1.5);
plot(1:n_max, MSE_prior2_gauss1, 'r-', 'LineWidth', 1.5);
plot(1:n_max, MSE_prior3_gauss1, 'g-', 'LineWidth', 1.5);
xlabel('Observations'); ylabel('MSE');
title('Gaussian: Estimating μ (σ known)');
legend('ML', 'Vague', 'Good Prior', 'Bad Prior', 'Location', 'best');
grid on;

subplot(3,2,4);
% Plot Posterior Evolution - basically the posterior density on the y axis
% and the actual mean on the x axis
mu_range = -2:0.01:12; % range for plotting
for i = 1:length(n_points)
    n = n_points(i);
    if n == 0
        % Just the prior
        mu_post = mu0_2; sigma_post = sigma0_2;
    else
        % Posterior - using precision weighting
        tau0 = 1/sigma0_2^2; % prior precision
        tau_data = n/sigma_known^2; % data precision
        tau_post = tau0 + tau_data; % posterior precision (precisions add!)
        % posterior mean is precision-weighted average
        mu_post = (tau0*mu0_2 + tau_data*mean(data_saved_gauss1(1:n))) / tau_post;
        sigma_post = sqrt(1/tau_post); % posterior std dev
    end
    plot(mu_range, normpdf(mu_range, mu_post, sigma_post), 'LineWidth', 1.5); hold on;
end
xline(mu_true, 'k--', 'LineWidth', 2);
xlabel('μ'); ylabel('Density');
title('Normal Posterior Evolution');
grid on;

% Gaussian Variance plots
subplot(3,2,5);
% Plot MSE Comparison - basically MSE on the y axis and the number of
% observations on the x axis
plot(1:n_max, MSE_ML_gauss2, 'k--', 'LineWidth', 2); hold on;
plot(1:n_max, MSE_prior1_gauss2, 'b-', 'LineWidth', 1.5);
plot(1:n_max, MSE_prior2_gauss2, 'r-', 'LineWidth', 1.5);
plot(1:n_max, MSE_prior3_gauss2, 'g-', 'LineWidth', 1.5);
xlabel('Observations'); ylabel('MSE');
title('Gaussian: Estimating σ² (μ known)');
legend('ML', 'Weak', 'Good Prior', 'Bad Prior', 'Location', 'best');
grid on;

subplot(3,2,6);
% Plot Posterior Evolution - basically the posterior density on the y axis
% and the actual variance on the x axis
sigma2_range = 0.01:0.01:12; % range for plotting (must be positive for variance)
for i = 1:length(n_points)
    n = n_points(i);
    if n == 0
        % Just the prior
        alpha_post = alpha2_ig; beta_post = beta2_ig;
    else
        % Posterior - using saved data
        S = sum((data_saved_gauss2(1:n) - mu_known).^2);
        alpha_post = alpha2_ig + n/2;
        beta_post = beta2_ig + S/2;
    end
    % Inverse-Gamma PDF calculation
    % IG(x; α, β) = (β^α / Γ(α)) * x^(-α-1) * exp(-β/x)
    pdf_vals = (beta_post^alpha_post / gamma(alpha_post)) .* ...
               sigma2_range.^(-alpha_post-1) .* exp(-beta_post./sigma2_range);
    plot(sigma2_range, pdf_vals, 'LineWidth', 1.5); hold on;
end
xline(sigma2_true, 'k--', 'LineWidth', 2);
xlabel('σ²'); ylabel('Density');
title('Inverse-Gamma Posterior Evolution');
xlim([0, 10]);
grid on;

sgtitle('All Three Conjugate Estimator Scenarios', 'FontSize', 16, 'FontWeight', 'bold');