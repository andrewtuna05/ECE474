import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from scipy.special import gamma

# Reproducibility
np.random.seed(0)

# Parameters
p_true = 0.7
m = 1
n_max = 100
n_mc = 1000

alpha1, beta1 = 1, 1   # Uniform prior
alpha2, beta2 = 7, 3   # Good prior
alpha3, beta3 = 2, 8   # Bad prior

MSE_ML_binom = np.zeros(n_max)
MSE_prior1_binom = np.zeros(n_max)
MSE_prior2_binom = np.zeros(n_max)
MSE_prior3_binom = np.zeros(n_max)

# Monte Carlo for Binomial
for trial in range(n_mc):
    data = np.random.binomial(1, p_true, n_max)

    if trial == 0:
        data_saved_binom = data.copy()

    for n in range(1, n_max + 1):
        successes = np.sum(data[:n])
        trials = n

        p_ML = successes / trials
        p_Bayes1 = (alpha1 + successes) / (alpha1 + beta1 + trials)
        p_Bayes2 = (alpha2 + successes) / (alpha2 + beta2 + trials)
        p_Bayes3 = (alpha3 + successes) / (alpha3 + beta3 + trials)

        MSE_ML_binom[n-1] += (p_ML - p_true) ** 2
        MSE_prior1_binom[n-1] += (p_Bayes1 - p_true) ** 2
        MSE_prior2_binom[n-1] += (p_Bayes2 - p_true) ** 2
        MSE_prior3_binom[n-1] += (p_Bayes3 - p_true) ** 2

MSE_ML_binom /= n_mc
MSE_prior1_binom /= n_mc
MSE_prior2_binom /= n_mc
MSE_prior3_binom /= n_mc

# Gaussian with known variance
mu_true = 5
sigma_known = 2

mu0_1, sigma0_1 = 0, 10
mu0_2, sigma0_2 = 5, 1
mu0_3, sigma0_3 = 10, 0.5

MSE_ML_gauss1 = np.zeros(n_max)
MSE_prior1_gauss1 = np.zeros(n_max)
MSE_prior2_gauss1 = np.zeros(n_max)
MSE_prior3_gauss1 = np.zeros(n_max)

for trial in range(n_mc):
    data = np.random.normal(mu_true, sigma_known, n_max)

    if trial == 0:
        data_saved_gauss1 = data.copy()

    for n in range(1, n_max + 1):
        mu_ML = np.mean(data[:n])
        tau_data = n / sigma_known**2

        tau0_1 = 1 / sigma0_1**2
        mu_Bayes1 = (tau0_1*mu0_1 + tau_data*np.mean(data[:n])) / (tau0_1 + tau_data)

        tau0_2 = 1 / sigma0_2**2
        mu_Bayes2 = (tau0_2*mu0_2 + tau_data*np.mean(data[:n])) / (tau0_2 + tau_data)

        tau0_3 = 1 / sigma0_3**2
        mu_Bayes3 = (tau0_3*mu0_3 + tau_data*np.mean(data[:n])) / (tau0_3 + tau_data)

        MSE_ML_gauss1[n-1] += (mu_ML - mu_true) ** 2
        MSE_prior1_gauss1[n-1] += (mu_Bayes1 - mu_true) ** 2
        MSE_prior2_gauss1[n-1] += (mu_Bayes2 - mu_true) ** 2
        MSE_prior3_gauss1[n-1] += (mu_Bayes3 - mu_true) ** 2

MSE_ML_gauss1 /= n_mc
MSE_prior1_gauss1 /= n_mc
MSE_prior2_gauss1 /= n_mc
MSE_prior3_gauss1 /= n_mc

# Gaussian with known mean
mu_known = 0
sigma2_true = 4

alpha1_ig, beta1_ig = 2, 2
alpha2_ig, beta2_ig = 5, 16
alpha3_ig, beta3_ig = 10, 10

MSE_ML_gauss2 = np.zeros(n_max)
MSE_prior1_gauss2 = np.zeros(n_max)
MSE_prior2_gauss2 = np.zeros(n_max)
MSE_prior3_gauss2 = np.zeros(n_max)

for trial in range(n_mc):
    data = np.random.normal(mu_known, np.sqrt(sigma2_true), n_max)

    if trial == 0:
        data_saved_gauss2 = data.copy()

    for n in range(1, n_max + 1):
        S = np.sum((data[:n] - mu_known)**2)
        sigma2_ML = S / n

        alpha_post1 = alpha1_ig + n/2
        beta_post1 = beta1_ig + S/2
        sigma2_Bayes1 = beta_post1 / (alpha_post1 - 1)

        alpha_post2 = alpha2_ig + n/2
        beta_post2 = beta2_ig + S/2
        sigma2_Bayes2 = beta_post2 / (alpha_post2 - 1)

        alpha_post3 = alpha3_ig + n/2
        beta_post3 = beta3_ig + S/2
        sigma2_Bayes3 = beta_post3 / (alpha_post3 - 1)

        MSE_ML_gauss2[n-1] += (sigma2_ML - sigma2_true) ** 2
        MSE_prior1_gauss2[n-1] += (sigma2_Bayes1 - sigma2_true) ** 2
        MSE_prior2_gauss2[n-1] += (sigma2_Bayes2 - sigma2_true) ** 2
        MSE_prior3_gauss2[n-1] += (sigma2_Bayes3 - sigma2_true) ** 2

MSE_ML_gauss2 /= n_mc
MSE_prior1_gauss2 /= n_mc
MSE_prior2_gauss2 /= n_mc
MSE_prior3_gauss2 /= n_mc

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(18, 9))

# Binomial MSE
axs[0,0].plot(MSE_ML_binom, 'k--', linewidth=2)
axs[0,0].plot(MSE_prior1_binom, 'b-', linewidth=1.5)
axs[0,0].plot(MSE_prior2_binom, 'r-', linewidth=1.5)
axs[0,0].plot(MSE_prior3_binom, 'g-', linewidth=1.5)
axs[0,0].set_xlabel("Observations")
axs[0,0].set_ylabel("MSE")
axs[0,0].set_title("Binomial: Estimating p")
axs[0,0].legend(["ML","Uniform","Good Prior","Bad Prior"])
axs[0,0].grid(True)

# Binomial posterior
n_points = [0, 5, 10, 25, 50, 100]
p_range = np.linspace(0,1,1000)
for n in n_points:
    if n == 0:
        a_post, b_post = alpha2, beta2
    else:
        s = np.sum(data_saved_binom[:n])
        a_post, b_post = alpha2 + s, beta2 + n*m - s
    axs[0,1].plot(p_range, beta.pdf(p_range, a_post, b_post), linewidth=1.5)
axs[0,1].axvline(p_true, color='k', linestyle='--', linewidth=2)
axs[0,1].set_xlabel("p")
axs[0,1].set_ylabel("Density")
axs[0,1].set_title("Beta Posterior Evolution")
axs[0,1].grid(True)

# Gaussian mean MSE
axs[1,0].plot(MSE_ML_gauss1, 'k--', linewidth=2)
axs[1,0].plot(MSE_prior1_gauss1, 'b-', linewidth=1.5)
axs[1,0].plot(MSE_prior2_gauss1, 'r-', linewidth=1.5)
axs[1,0].plot(MSE_prior3_gauss1, 'g-', linewidth=1.5)
axs[1,0].set_xlabel("Observations")
axs[1,0].set_ylabel("MSE")
axs[1,0].set_title("Gaussian: Estimating μ (σ known)")
axs[1,0].legend(["ML","Vague","Good Prior","Bad Prior"])
axs[1,0].grid(True)

# Gaussian mean posterior
mu_range = np.linspace(-2,12,1400)
for n in n_points:
    if n == 0:
        mu_post, sigma_post = mu0_2, sigma0_2
    else:
        tau0 = 1/sigma0_2**2
        tau_data = n/sigma_known**2
        tau_post = tau0 + tau_data
        mu_post = (tau0*mu0_2 + tau_data*np.mean(data_saved_gauss1[:n])) / tau_post
        sigma_post = np.sqrt(1/tau_post)
    axs[1,1].plot(mu_range, norm.pdf(mu_range, mu_post, sigma_post), linewidth=1.5)
axs[1,1].axvline(mu_true, color='k', linestyle='--', linewidth=2)
axs[1,1].set_xlabel("μ")
axs[1,1].set_ylabel("Density")
axs[1,1].set_title("Normal Posterior Evolution")
axs[1,1].grid(True)

# Gaussian variance MSE
axs[2,0].plot(MSE_ML_gauss2, 'k--', linewidth=2)
axs[2,0].plot(MSE_prior1_gauss2, 'b-', linewidth=1.5)
axs[2,0].plot(MSE_prior2_gauss2, 'r-', linewidth=1.5)
axs[2,0].plot(MSE_prior3_gauss2, 'g-', linewidth=1.5)
axs[2,0].set_xlabel("Observations")
axs[2,0].set_ylabel("MSE")
axs[2,0].set_title("Gaussian: Estimating σ² (μ known)")
axs[2,0].legend(["ML","Weak","Good Prior","Bad Prior"])
axs[2,0].grid(True)

# Gaussian variance posterior
sigma2_range = np.linspace(0.01,12,1000)
for n in n_points:
    if n == 0:
        a_post, b_post = alpha2_ig, beta2_ig
    else:
        S = np.sum((data_saved_gauss2[:n] - mu_known)**2)
        a_post = alpha2_ig + n/2
        b_post = beta2_ig + S/2
    pdf_vals = (b_post**a_post / gamma(a_post)) * sigma2_range**(-a_post-1) * np.exp(-b_post/sigma2_range)
    axs[2,1].plot(sigma2_range, pdf_vals, linewidth=1.5)
axs[2,1].axvline(sigma2_true, color='k', linestyle='--', linewidth=2)
axs[2,1].set_xlabel("σ²")
axs[2,1].set_ylabel("Density")
axs[2,1].set_title("Inverse-Gamma Posterior Evolution")
axs[2,1].set_xlim([0,10])
axs[2,1].grid(True)

plt.suptitle("All Three Conjugate Estimator Scenarios", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()
