# BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference

### *Ivo Brink, Cile van Marken, Sebastiaan Snel, Liang Telkamp, Jesse Wiers*

*May 2024*

---

In this blogpost we discuss our findings on enhancing the quality of images generated by diffusion models by incorporating pixel-wise uncertainty estimation through Bayesian inference. This involves reproducing the research of the paper *["BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference"](https://arxiv.org/abs/2310.11142)* by Kou, Gan, Wang, Li, and Deng (2024) and suggesting a methodology to improve their Bayesian inference implementation. The paper discusses the effectiveness of diffusion models (DMs) in transforming noise vectors into natural images across various applications, such as image synthesis and text-to-image generation. Despite their capabilities, these models often produce low-quality images, leading to a poor user experience. 

> HIER NOG EVEN ONDERBOUWEN WAT ER MINDER GOED IS -> BEWIJS MET REFERENCE

In addition, the paper highlights the challenge of filtering out these low-quality images due to the lack of a reliable metric for assessing image quality, with traditional evaluation metrics. Often used evaluation metrics are FID and Inception Scores which are not perfect to evaluate the generated images, since they fall short in capturing nuanced aspects of image quality. This inherent shortcoming prompts the search for more effective methods to enhance DMs, aiming to improve diversity and rectify artifacts in generated images. Our aim is to address these shortcomings by introducing pixel-wise uncertainty as a means to improve DMs. By incorporating uncertainty estimation, the model can identify and potentially discard the most uncertain images, or even improve generated images and enhancing diversity in the generated samples. *Figure 1* points out the architecture that is used in conducting the "Bayesdiff" research paper. 

<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/intro_00.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b>  Given an initial point $x_T \sim \mathcal{N}(0,I)$, the BayesDiff framework incorporates uncertainty into the denoising process and generates images with pixel-wise uncertainty estimates.</td>
  </tr>
</table>

## TL;DR
> TODO


---

## How do diffusion models work?

Diffusion models are a family of probabilistic generative models that progressively destruct data by injecting noise, then learn to reverse this process for sample generation (Yang et al., 2023). This forward process, parameterizd by $q$ in the left equation, uses datapoints $x_0 \sim q(x)$, sampled from a real data distribution in which a small ammount of Gaussian noise, with a variance of $\beta_t \in (0,1)$, is added in $T$ steps. This results in a sequence of noisy samples $x_1,...,x_T$ parameterized by the right equation below. 

$$\begin{align} 
q\left( x_1, \ldots, x_T \mid x_0 \right) := \prod_{t=1}^T q \left( x_t \mid x_{t-1} \right) & \qquad \qquad 
q\left( x_t \mid x_{t-1} \right) := \mathcal{N}\left( x_t ; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I} \right) & \qquad \qquad
\end{align}$$

As step $t$ becomes larger the data sample $x_0$ gradually loses its distinguishable features and becomes equivalent to an isotrophic Gaussian function namely, a noisy image. *Figure 2* points out both the forward diffusion process that gradually adds noise to the image as well as the reverse process. In this reversed process, the true sample from a Gaussian noise input $x_T \sim \mathcal{N}(0,I)$ is recreated by sampling from $q(x_{t-1}|x_t)$. Sampling from $q(x_{t-1}|x_t)$ is hard because the entire dataset needs to be used. The goal is to learn model $p_{\theta}$, parameterized by the next equation.

$$p_\theta \left( x_{t-1} \mid x_t \right) := \mathcal{N} \left( x_{t-1} ; \mu_\theta \left( x_t, t \right), \Sigma_\theta \left( x_t, t \right) \right) \qquad \qquad$$

<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/DDPM.png" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 2.</b>  The Markov chain of forward (and reverse) diffusion process (Ho et al. 2020).</td>
  </tr>
</table>

Building upon DDPMs, Song et al. (2020) propose a different approach namely, Denoising Diffusion Implicit Models (DDIMs), which is the main DM that is used in this blogpost. DDIMs provide a more efficient class of iterative implicit probabilistic models with the same training procedure. In contrast to DDPM, DDIM yields an equal marginal noise distribution but deterministically maps noise back to the original data samples. This makes it possible to train the diffusion model up to any arbitrary number of forward steps but only sample from a subset of steps in the generative process. DDIM yields the following advantages compared to DDPM:

- The ability to generate higher-quality samples using a much fewer number of steps.
- The “consistency” property since the generative process is deterministic. This means that multiple samples conditioned on the same latent variable should have similar high-level features.
- DDIM can do semantically meaningful interpolation in the latent variable because of the "consistency".

---

## What is the role of Bayesian inference here?

#### Laplace & Hessian

Bayesian inference can be used for uncertainty quantification in Diffusion Models by turning a deterministic neural network into a Bayesian Neural Network (BNN). One such Bayesian approach that is used in the "BayesDiff" paper is Last Layer Laplace Approximation (LLLA)(Daxberger et al., 2022). This approach can be applied to models post-hoc and is very cost-efficient. In order to get the posterior of the weight of the last layer, LLLA is performed by a Laplace approximation of the weight of the last layer $\theta$, while assuming the previous layer to be fixed. Let us denote $p(\theta|\mathcal{D}) = \mathcal{N}(\theta|\theta_{MAP}, H^{-1})$ where $H$ is the Hessian of the negative log-posterior w.r.t. $\theta$ at $\theta_{MAP}$ (Kristiadi et al., 2020). 

LLLA approximates the predictive posterior using a Gaussian distribution centered at a local maximum denoted by $\theta_{MAP}$ and a covariance matrix corresponding to the local curvature. This covariance matrix is computed by approximating the inverse of the Hessian denoted by $H^{-1}$. Using the variance of the predictive posterior, the pixel-wise uncertainty can be computed. In the context of our research, LLLA is incorporated into the noise prediction model in DMs for uncertainty measurements at a single timestep. The noise prediction model is trained to minimize the next equation $p$ under a weight decay reguralizer that corresponds to the Gaussian prior on the NN parameters. Emphasized by the "Bayesdiff" paper, the Gaussian approximate posterior distribution on the parameters directly leads to a Gaussian posterior predictive: 

$$\begin{align} 
p\left( \epsilon_t \mid x_t, t, \mathcal{D} \right) \approx \mathcal{N} \left( \epsilon_{\theta}(x_t, t), diag(\gamma^2_{\theta}(x_t,t)) \right) & \qquad \qquad 
\end{align}$$


#### Hessian free Laplace
In conducting our research we propose the Hessian-free Laplace (HFL) approach (McInerney & Kallus, 2024)  as an alternative to the diagonal Hessian that is used in the "BayesDiff" paper. To make their research computationally less heavy, the authors make use of diagonal factorization, ignoring all off-diagonal elements of the Hessian. However, diagonal approximations of the Hessian are outperformed significantly by the KFAC approach, which offers greater expressiveness as mentioned by (Daxberger et al., 2022) in exchange for more computation. As this might enhance the resulting uncertainty maps, we propose replacing diagonal factorization of the Hessian with KFAC factorization. 
> Remove KFAC???

Additionally, (McInerney & Kallus, 2024) demonstrate that computing the Hessian might not even be necessary using the Hessian-free Laplace (HFL). Should the HFL lead to similar results as the LLLA, computing the pixel-wise uncertainty could be achieved with significantly less computation. In contrast to original Hessian approach, HFL uses the curvature of both the log posterior and network prediction to estimate its variance. McInerney and Kallus prove that HFL yields the same variance as LA which results in equal performance of the HFL compared to that of exact and approximate Hessians. Emphasizing the HFL architecture we point out its pseudocode.

<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/HFL_pseudo.png" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 3.</b>  Hessian-free Laplace in pseudocode (McInerney and Kallus, 2024).</td>
  </tr>
</table>


The motivation for this proposal, and to not use LA, is the computational bottleneck of LA which is the step of calculating and inverting the Hessian matrix $H$ of the log posterior.



---

## How do we evaluate Diffusion Models?
As in many ML problems, a robust evaluation metric is key for examining results. Unfortunately, we find that the Frechet Inception Distance (FID) may react different in some cases than the gold standard, human raters. Therefore, we propose to evaluate the images with three different evaluation metrics namely,"Inception score" (IS), "Frechet Inception Distance" (FID) and "CLIP embeddings Maximum Mean Discrepancy" (CMMD). Despite, each of this metrics containing advantages and disadvantages in capturing the uncertainty of generated images, we propose to compare their results. 

### IS
The Inception Score (IS) measures variety in images and that each image distinctly looks like something. A higher IS means a better image quality (Barratt & Sharma, 2018). IS has been used in the field of generative modeling as a benchmark for assessing the performance of GANs and related algorithms (Salimans et al., 2016). For general application IS is denoted by,

$$\begin{align} 
\text{IS} = \exp(\mathbb{E}_x[KL(p(y|x) || p(y))])
\end{align}$$

where $\mathbb{E}_x$ represents the expectation over images $x$ generated by the model, $p(y|x)$ is the conditional class distribution given the generated image $x$ and $p(y)$ is the marginal class distribution. Despite its wide usage, IS has the disadvantage to not use statistics of real world samples in comparison with the synthetic/generated images. Therefore, Heusel et al. propose FID that overcomes this disadvantage. 


### FID
FID is a widely used evaluation metric for assessing the quality of generated images compared to real ones (Heusel et al., 2018). It quantifies the dissimilarity between the distributions of features extracted from real images and those generated by an algorithm. Let us examin the key components of FID, where $m$ and $m_w$ are the means of the feature representations of the real and generated images respectively, $C$ and $C_w$ are the covariance matrices of the feature representations of the real and generated images.

$$\begin{align} 
d^2((m, C), (m_w, C_w)) = || m - m_w ||_2^2 + \text{Tr}(C + C_w - 2(C C_w)^{1/2})
\end{align}$$

Recent research has highlighted limitations of FID, particularly its inconsistency with human perception in certain cases. For instance, FID may inaccurately assess image quality under progressive distortions, leading to misleading results (Jayasumana et al., 2024). As can be seen, Figure 4 emphasizes their main findings in which FID does not reflect progressive distortion applied to images. To adress these shortcomings Jayasuma et al. propose a different metric namely CMMD.

<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/CMMD.png" width=500></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 4.</b>  Behaviour of FID and CMMD under distortions. CMMD increases with a higher distortion level, identifying the degradation in image quality with increasing distortions. FID improves (goes down) for the first few distortion levels, suggesting that quality improves when these more subtle distortions are applied (Jayasumana et al., 2024).</td>
  </tr>
</table>


### CMMD
We want to utilize CMMD, which is not dependent on the sample size, as a more robust metric, suitable for evaluation of our generated images (Jayasumana et al., 2024).


---

## Reproduction of the experiments

**Data**

**Implementation**


---

## Ablation study
### Hyperparameter tuning

First, we propose to do hyperparameter tuning on the LLLA. The code provided with the paper shows that the authors have fixed the parameters for the prior precision and the prior mean of the LLLA on 1 and 0, respectively. These two hyperparameters influence the behavior of the LLLA and should thus be fine-tuned. In the case that changing these hyperparameters doesn’t have a large impact on the resulting uncertainty maps, the question can be raised as to if a Bayesian approach to uncertainty in DMs is a good one.

The Bayesian approach to uncertainty in generated images is an interesting one. Before we can extend on this idea, it is key to check whether this approach is a viable one. The Last Layer Laplacian, which is used for the uncertainty estimation, is computed using an approximation of the Hessian matrix. The authors have chosen to do a diagonal approximation of the Hessian for this. Additionally, they have fixed the following parameters of this approximation on the following values: sigma_noise: 1, prior_precision: 1. 
However, the authors have not specified why these specific values are used. The authors elaborate: “In our experiment, we just set both the noise and the precision as the default value 1 used in Laplace library in our experiment. Empirical results show that the pixel-wise variance of image do not change much when we adjust the value of prior precision. We have not tested the sensitivity to the choice of observation noise yet and will experiment with this interesting question in our future work.” 
However, one would expect that the uncertainty maps would differ, when these parameters are changed. When this is not the case, one could raise the question as to if the Bayesian approach is a valid one.

We ran a hyperparameter search over the variables sigma_noise (0-1) and prior_precision (0-1000). The results are shown below. The uncertainty maps show miniscule changes when different parameters are used. One would expect that uncertainty maps with a higher precision would be very dark compared to uncertainty maps with a lower precision, as the model is more certain with the strong prior.
<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/hyperparameter_DDIM_guided.jpg" width=600></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 4.</b>  Hyperparameter tuning on the DDIM model.</td>
  </tr>
</table>


<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/big_birds.png" width=1200></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 5.</b>  Uncertainty maps (EVEN SETTINGS TOEVOEGEN) </td>
  </tr>
</table>

<table align="center">
  <tr align="center">
      <td><img src="https://github.com/cilevanmarken/BayesDiff/raw/main/timestep_uncertainty.jpg" width=700></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 5.</b>  Uncertainty maps (EVEN SETTINGS TOEVOEGEN) </td>
  </tr>
</table>

## Alternatives to Bayesian uncertainty
> Here we will discuss non-bayesian methods that simulate results from the paper

## How to aggregate?
> Here we test different aggregation methods

The availability of the uncertainty estimates per
pixel in the generated images presents the need for an effective aggregation
method. This method plays a significant role in deriving the final evalua-
tion metric. The original paper suggests a sum across all pixel uncertainties.
Nonetheless, this mere statistical interpretation might not fully capture the role
different image regions play in assessing the realism of an image. Therefore, we
suggest using a combination of weighted aggregation with segmentation maps,
this might highlight uncertainties within primary objects and neglect uncer-
tainties in complex and cluttered backgrounds. The segmentation maps may be
derived through pre-trained models or simple downsampling operations such as
pooling. Moreover, other statistical interpretations, such as variance and per-
centiles, might also be analyzed to see to what extent they describe the realism
of images

## Final remarks





## 



## Authors' Contributions
- **Literature review:** Sebastiaan will be taking charge over the literature review, but every member contributes to it.
- **Writing the general parts of the tutorial:** Sebastiaan \& Liang will be involved with the general sections of the tutorial, such as introduction, background, relevancy, etc.
- **Reproduce the findings:** Jesse will work on reproducing the qualitative and quantitative results of the paper.
- **Hyperparameter tuning:** Cile will be taking charge in tuning the hyperparameters of the Laplacian, therefore she will also write the part of the background, experiment setup, results, etc.
- **Replace the Hessian by KFAC \& Making the Laplacian Hessian-Free:** This is one of our extensions, Cile and Ivo will be looking into this part.
- **Implement a variant of the aggregation mechanism:** Ivo will work on applying a variety of aggregation methods on the uncertainty estimates.
- **Replace the evaluation metrics:** Liang will do further research on different metrics and will report results on replacing the FID with the CMMD.
- Writing the tutorial will divided amongst all members, since the tutorial is highly dependent on the execution of the code. Each member that engaged in a specific experiment will write that part of the tutorial.


## References

[Barratt and Sharma, 2018] Barratt, S., & Sharma, R. (2018). A Note on the Inception Score. arXiv preprint arXiv:1801.01973

[Chen et al., 2014] Chen, T., Fox, E., & Guestrin, C. (2014). Stochastic Gradient Hamiltonian Monte Carlo. In Proceedings of the 31st International Conference on Machine Learning (pp. 1683-1691). PMLR. http://proceedings.mlr.press/v32/cheni14.pdf

[Daxberger et al., 2022] Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2022). Laplace Redux -- Effortless Bayesian Deep Learning. arXiv preprint arXiv:2106.14806

[Heusel et al., 2018] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2018). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. arXiv preprint arXiv:1706.08500

[Ho et al., 2020] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2006.11239

[Jayasumana et al., 2024] Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., & Kumar, S. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation. arXiv preprint arXiv:2401.09603

[Kou et al., 2024] Kou, S., Gan, L., Wang, D., Li, C., & Deng, Z. (2024). BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference. arXiv preprint arXiv:2310.11142

[Kristiadi et al., 2020] Kristiadi, A., Hein, M., & Hennig, P. (2020). Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks. arXiv preprint arXiv:2002.10118.

[McInerney and Kallus, 2024] McInerney, J., & Kallus, N. (2024). Hessian-Free Laplace in Bayesian Deep Learning. arXiv preprint arXiv:2403.10671

[Obukhov and Krasnyanskiy, 2020] Obukhov, A., & Krasnyanskiy, M. (2020). Quality assessment method for GAN based on modified metrics inception score and Fréchet inception distance. In Software Engineering Perspectives in Intelligent Systems: Proceedings of 4th Computational Methods in Systems and Software 2020, Vol. 1 4 (pp. 102-114). Springer.

[Sauer et al., 2023] Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042

[Salimans et al., 2016] Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs. arXiv preprint arXiv:1606.03498

[Song et al., 2020] Song, J., Meng, C., & Ermon, S. (2022). Denoising Diffusion Implicit Models. arXiv preprint arXiv:2010.02502

[Wenzel et al., 2020] Wenzel, F., Roth, K., Veeling, B. S., Świątkowski, J., Tran, L., Mandt, S., Snoek, J., Jenatton, R., & Nowozin, S. (2020). How Good is the Bayes Posterior in Deep Neural Networks Really? arXiv preprint arXiv:2002.02405

[Yang et al., 2023] Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., & Yang, M.-H. (2023). Diffusion Models: A Comprehensive Survey of Methods and Applications. ACM Computing Surveys, 56(4), 105. https://doi.org/10.1145/3626235

[Zhdanov et al., 2023] Zhdanov, M., Dereka, S., & Kolesnikov, S. (2023). Unveiling Empirical Pathologies of Laplace Approximation for Uncertainty Estimation. arXiv preprint arXiv:2312.10464








