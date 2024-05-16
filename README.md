# BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference

### *Ivo Brink, Cile van Marken, Sebastiaan Snel, Liang Telkamp, Jesse Wiers*

*May 2024*

---

In this blogpost we discuss our findings on enhancing the quality of images generated by diffusion models by incorporating pixel-wise uncertainty estimation through Bayesian inference. This involves reproducing the research of the paper *["BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference"](https://arxiv.org/abs/2310.11142)* by Kou, Gan, Wang, Li, and Deng (2024) and suggesting a methodology to improve their Bayesian inference implementation. The paper discusses the effectiveness of diffusion models (DMs) in transforming noise vectors into natural images across various applications, such as image synthesis and text-to-image generation. Despite their capabilities, these models often produce low-quality images, leading to a poor user experience. The paper highlights the challenge of filtering out these low-quality images due to the lack of a reliable metric for assessing image quality, with traditional evaluation metrics.

---

## Introduction

Diffusion Models have shown to be capable of generating realistic samples across diverse domains, but are also prone to producing low-quality images. Evaluation metrics such as FID and Inception Scores are not perfect to evaluate generated images, since they fall short in capturing nuanced aspects of image quality. This inherent shortcoming prompts the search for more effective methods to enhance DMs, aiming to improve diversity and rectify artifacts in generated images. This paper tries to address these shortcomings by introducing pixel-wise uncertainty as a means to improve DMs. By incorporating uncertainty estimation, the model can identify and potentially discard the most uncertain images, or even improve generated images and enhancing diversity in the generated samples.


<img src="https://github.com/cilevanmarken/BayesDiff/raw/main/intro_00.png" alt="Alt text" width="600"/>








### How do diffusion models work?

Diffusion models are a family of probabilistic generative models that progressively destruct data by injecting noise, then learn to reverse this process for sample generation (Yang et al., 2023). First, a Denoising Diffusion Probabilistic Model (DDPM), which is the original diffusion approach, makes use of two Markov chains: a forward chain that perturbs data to noise, and a reverse chain that converts noise back to data. Building upon DDPMs, Song, Meng, and Ermon (2022) proposed a different approach namely, Denoising Diffusion Implicit Models (DDIMs). DDIMs provides a more efficient class of iterative implicit probabilistic models with the same training procedure.

### Bayesian inference

### Laplace & Hessian

Bayesian approaches can be used for uncertainty quantification in Diffusion Models. One such Bayesian approach is Last Layer Laplace Approximation (LLLA), which can be applied to models post-hoc and is very cost-efficient. LLLA approximates the predictive posterior using a Gaussian distribution centered at a local maximum (θMAP) and a covariance matrix corresponding to the local curvature. This covariance matrix is computed by approximating the inverse of the Hessian. Using the variance of the predictive posterior, the pixel-wise uncertainty can be computed (Daxberger et al., 2022).

### Hessian free Laplace

### Evaluating images 

The Frechet Inception Distance (FID) is a widely used evaluation metric for assessing the quality of generated images compared to real ones. It quantifies the dissimilarity between the distributions of features extracted from real images and those generated by an algorithm. However, recent research has highlighted limitations of FID, particularly its inconsistency with human perception in certain cases. For instance, FID may inaccurately assess image quality under progressive distortions, leading to misleading results (Jayasumana et al., 2024). The Inception Score (IS) measures variety in images and that each image distinctly looks like something. A higher IS means a better image quality. (Barratt & Sharma, 2018) The Inception Score has been used in the field of generative modeling as a benchmark for assessing the performance of GANs and related algorithms (Salimans et al., 2016).

## Methodology

#### *Hyperparameter tuning:* 

First, we propose to do hyperparameter tuning on the LLLA. The code provided with the paper shows that the authors have fixed the parameters for the prior precision and the prior mean of the LLLA on 1 and 0, respectively. These two hyperparameters influence the behavior of the LLLA and should thus be fine-tuned. In the case that changing these hyperparameters doesn’t have a large impact on the resulting uncertainty maps, the question can be raised as to if a Bayesian approach to uncertainty in DMs is a good one.

#### *Approximating the Hessian:* 

A computational bottleneck of LLLA lies in computing the inverted Hessian. To make this computationally less heavy, the authors make use of diagonal factorization, ignoring all off-diagonal elements of the Hessian. However, diagonal approximations of the Hessian are outperformed significantly by the Kronecker-Factored Approximation Curvature (KFAC) approach, which offers greater expressiveness as mentioned by (Daxberger et al., 2022). As this might enhance the resulting uncertainty maps, we propose replacing diagonal factorization of the Hessian with KFAC factorization. Additionally, (McInerney & Kallus, 2024) demonstrates that computing the Hessian might not even be necessary using the Hessian-free Laplace (HFL). Should the HFL lead to similar results as the LLLA, computing the pixel-wise uncertainty could be achieved with significantly less computation.

#### *Aggregation mechanism:*

The availability of the uncertainty estimates per pixel in the generated images presents the need for an effective aggregation method. This method plays a significant role in deriving the final evaluation metric. The original paper suggests a sum across all pixel uncertainties. Nonetheless, this mere statistical interpretation might not fully capture the role different image regions play in assessing the realism of an image. Therefore, we suggest using a combination of weighted aggregation with segmentation maps, this might highlight uncertainties within primary objects and neglect uncertainties in complex and cluttered backgrounds. The segmentation maps may be derived through pre-trained models or simple downsampling operations such as pooling. Moreover, other statistical interpretations, such as variance and percentiles, might also be analyzed to see to what extent they describe the realism of images.

#### *FID metric:*

One of the most popular evaluation metrics for evaluating generated images is FID. FID estimates the distance between a distribution of features of real images, and those of images generated by some model. We want to utilize CMMD, which is not dependent on the sample size, as a more robust metric, suitable for evaluation of our generated images (Jayasumana et al., 2024).

#### *CMMD metric:*


## Reproduction of the experiments

**Data**

**Implementation**


## Results and Analysis


### Stable diffusion


## Samplers:

### DDIM

### DDPM

### DPM

### 




## Conclusion



## Contributions
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

[Ho et al., 2020] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. arXiv preprint arXiv:2006.11239

[Jayasumana et al., 2024] Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., & Kumar, S. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation. arXiv preprint arXiv:2401.09603

[Kou et al., 2024] Kou, S., Gan, L., Wang, D., Li, C., & Deng, Z. (2024). BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference. arXiv preprint arXiv:2310.11142

[McInerney and Kallus, 2024] McInerney, J., & Kallus, N. (2024). Hessian-Free Laplace in Bayesian Deep Learning. arXiv preprint arXiv:2403.10671

[Obukhov and Krasnyanskiy, 2020] Obukhov, A., & Krasnyanskiy, M. (2020). Quality assessment method for GAN based on modified metrics inception score and Fréchet inception distance. In Software Engineering Perspectives in Intelligent Systems: Proceedings of 4th Computational Methods in Systems and Software 2020, Vol. 1 4 (pp. 102-114). Springer.

[Sauer et al., 2023] Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). Adversarial diffusion distillation. arXiv preprint arXiv:2311.17042

[Salimans et al., 2016] Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs. arXiv preprint arXiv:1606.03498

[Song et al., 2022] Song, J., Meng, C., & Ermon, S. (2022). Denoising Diffusion Implicit Models. arXiv preprint arXiv:2010.02502

[Wenzel et al., 2020] Wenzel, F., Roth, K., Veeling, B. S., Świątkowski, J., Tran, L., Mandt, S., Snoek, J., Jenatton, R., & Nowozin, S. (2020). How Good is the Bayes Posterior in Deep Neural Networks Really? arXiv preprint arXiv:2002.02405

[Yang et al., 2023] Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., & Yang, M.-H. (2023). Diffusion Models: A Comprehensive Survey of Methods and Applications. ACM Computing Surveys, 56(4), 105. https://doi.org/10.1145/3626235

[Zhdanov et al., 2023] Zhdanov, M., Dereka, S., & Kolesnikov, S. (2023). Unveiling Empirical Pathologies of Laplace Approximation for Uncertainty Estimation. arXiv preprint arXiv:2312.10464







