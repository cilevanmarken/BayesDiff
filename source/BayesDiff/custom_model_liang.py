import torch.nn as nn
import torch
from laplace.baselaplace import DiagLaplace
from laplace.curvature.backpack import BackPackEF
from torch.nn.utils import parameters_to_vector
import copy

class CustomModel(nn.Module):
    def __init__(self, diff_model, dataloader, args, config):
        super().__init__()
        self.args = args
        self.config = config
        print('Custom Model Liang')
        if self.config.data.dataset == "CELEBA":
            self.conv_out = diff_model.conv_out
            self.copied_cov_out = copy.deepcopy(self.conv_out)

            self.feature_extractor = diff_model
            self.feature_extractor.conv_out = nn.Identity()
            
            self.conv_out_la = DiagLaplace(nn.Sequential(self.conv_out, nn.Flatten(1, -1)), likelihood='regression', 
                                    sigma_noise=self.args.sigma_noise, prior_precision=self.args.prior_precision, prior_mean=0.0, temperature=1.0,
                                    backend=BackPackEF)
            self.fit(dataloader)
        else:
            self.conv_out = diff_model.out[2]
            self.copied_cov_out = copy.deepcopy(self.conv_out)

            self.feature_extractor = diff_model
            self.feature_extractor.out[2] = nn.Identity()

            self.conv_out_la = DiagLaplace(nn.Sequential(self.conv_out, nn.Flatten(1, -1)), likelihood='regression', 
                                    sigma_noise=self.args.sigma_noise, prior_precision=self.args.prior_precision, prior_mean=0.0, temperature=1.0,
                                    backend=BackPackEF)
            self.fit(dataloader)

    def fit(self, train_loader, override=True):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        """
        config = self.config
        print(f'Prior precision {self.conv_out_la.prior_precision}')
        if self.config.data.dataset == "CELEBA":
            if override:
                self.conv_out_la._init_H()
                self.conv_out_la.n_data = 0

            self.conv_out_la.model.eval()
            self.conv_out_la.mean = parameters_to_vector(self.conv_out_la.model.parameters()).detach()

            N = len(train_loader.dataset)
            self.conv_out_la.H += torch.full((self.conv_out_la.H.shape[0],), self.conv_out_la.prior_precision[0] * N).to(self.conv_out_la._device) # Add for batch size diagonal of identity matrix (prior precision not needed becauase equal to 1)

            self.conv_out_la.n_data += N

        else: 
            if override:
                self.conv_out_la._init_H()
                self.conv_out_la.n_data = 0

            self.conv_out_la.model.eval()
            self.conv_out_la.mean = parameters_to_vector(self.conv_out_la.model.parameters()).detach()

            N = len(train_loader.dataset) # 250
            self.conv_out_la.H += torch.full((self.conv_out_la.H.shape[0],), self.conv_out_la.prior_precision[0] * N).to(self.conv_out_la._device) # Add for batch size diagonal of identity matrix (prior precision not needed becauase equal to 1)

            self.conv_out_la.n_data += N

    def forward(self, x, t, **model_kwargs):

        if self.config.data.dataset == "CELEBA":
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t)
            # #### glm predict
            # f_mean, f_cov = self.conv_out_la(x, pred_type='glm')
            # return f_mean, torch.diagonal(f_cov, dim1=1, dim2=-1)
            #### nn predict
            mean, var = self.conv_out_la(x, pred_type='nn', link_approx='mc', n_samples=100)
            mean = torch.reshape(mean, (-1, 3, self.config.data.image_size, self.config.data.image_size))
            var = torch.reshape(var, (-1, 3, self.config.data.image_size, self.config.data.image_size))
            return (mean, var)

        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t, **model_kwargs)
            # #### glm predict
            # f_mean, f_cov= self.conv_out_la(x, pred_type='glm')
            # f_mean = torch.reshape(f_mean, (-1, 6, self.config.data.image_size, self.config.data.image_size))
            # f_var = torch.reshape(torch.diagonal(f_cov,dim1=1, dim2=-1), (-1, 6, self.config.data.image_size, self.config.data.image_size))

            #### nn predict
            f_mean, f_var= self.conv_out_la(x, pred_type='nn', link_approx='mc', n_samples=100)
            f_mean = torch.reshape(f_mean, (-1, 6, self.config.data.image_size, self.config.data.image_size))
            f_var = torch.reshape(f_var, (-1, 6, self.config.data.image_size, self.config.data.image_size))

            f_mean = torch.split(f_mean, 3, dim=1)[0]
            f_var = torch.split(f_var, 3, dim=1)[0]
            return (f_mean, f_var)
    
    def accurate_forward(self, x, t, **model_kwargs):
        
        if self.config.data.dataset == "CELEBA":
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t)
                acc_mean = self.copied_cov_out(x)
                
            acc_mean = torch.reshape(acc_mean, (-1, 3, self.config.data.image_size, self.config.data.image_size))
            
            return acc_mean

        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                x = self.feature_extractor(x, t, **model_kwargs)
                acc_mean = self.copied_cov_out(x)

            acc_mean = torch.reshape(acc_mean, (-1, 6, self.config.data.image_size, self.config.data.image_size))
            acc_mean = torch.split(acc_mean, 3, dim=1)[0]          
            
            return acc_mean