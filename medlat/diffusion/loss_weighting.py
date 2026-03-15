import torch 
from abc import ABC, abstractmethod


class BaseLossWeighting(ABC, torch.nn.Module):
    """
    Defines the abstract class for all the loss weighting classes.
    """
    def __init__(self, objective = "eps"):
        super().__init__()

    @abstractmethod
    def forward(self, timesteps, alphas_cumprod):
        pass

class NoneWeighting(BaseLossWeighting):
    def __init__(self, objective = "eps", *args, **kwargs):
        super().__init__(objective)
        self.objective = objective

    def forward(self, timesteps, alphas_cumprod):
        """
        Calculates the forward pass of the model.

        Args:
            timesteps (torch.Tensor): A tensor of shape (B,) containing the timesteps.
            alphas_cumprod (torch.Tensor): A tensor of shape (T,) containing the cumulative product of alphas.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the weighted SNR values.
        """
        return torch.ones_like(timesteps)

class SNRWeighting(BaseLossWeighting):
    def __init__(self, objective = "eps", *args, **kwargs):
        """
        Initializes an instance of the SNRWEighting class with the specified objective.

        :param objective: (str) The objective of the class. Defaults to "noise".
        :return: None
        """
        super().__init__(objective)
        self.objective = objective

    def forward(self, timesteps, alphas_cumprod):
        """
        Calculates the forward pass of the model.

        Args:
            timesteps (torch.Tensor): A tensor of shape (B,) containing the timesteps.
            alphas_cumprod (torch.Tensor): A tensor of shape (T,) containing the cumulative product of alphas.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the weighted SNR values.
        """

        if self.objective == "eps":
            return torch.ones_like(timesteps)
        elif self.objective == "x0":
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)[timesteps]
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)[timesteps]

            snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
            return snr
        else:
            raise NotImplementedError
        
class InverseSNRWeighting(BaseLossWeighting):
    def __init__(self, objective = "eps", *args, **kwargs):
        """
        Initializes an instance of the SNRWEighting class with the specified objective.

        :param objective: (str) The objective of the class. Defaults to "noise".
        :return: None
        """
        super().__init__(objective)
        self.objective = objective

    def forward(self, timesteps, alphas_cumprod):
        """
        Calculates the forward pass of the model.

        Args:
            timesteps (torch.Tensor): A tensor of shape (B,) containing the timesteps.
            alphas_cumprod (torch.Tensor): A tensor of shape (T,) containing the cumulative product of alphas.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the weighted SNR values.
        """


        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)[timesteps]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)[timesteps]

        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return 1 / snr


class EqualWeighting(BaseLossWeighting):
    def __init__(self, objective = "eps", *args, **kwargs):
        """
        Initializes an instance of the SNRWEighting class with the specified objective.

        :param objective: (str) The objective of the class. Defaults to "noise".
        :return: None
        """
        super().__init__(objective)
        self.objective = objective

    def forward(self, timesteps, alphas_cumprod):
        """
        Calculates the forward pass of the model.

        Args:
            timesteps (torch.Tensor): A tensor of shape (B,) containing the timesteps.
            alphas_cumprod (torch.Tensor): A tensor of shape (T,) containing the cumulative product of alphas.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the weighted SNR values.
        """


        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)[timesteps]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)[timesteps]

        scale = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod)
        return 1 / scale
    
class MinSNRWeighting(BaseLossWeighting):
    def __init__(self, objective = "eps", snr_gamma = 5.0):
        """
        Initializes an instance of the MinSNRWeighting class.
        https://arxiv.org/pdf/2303.09556
        https://github.com/TiankaiHang/Min-SNR-Diffusion-Training

        Args:
            objective (str, optional): The objective of the weighting scheme. Defaults to "noise".
            snr_gamma (float, optional): The gamma value for the SNR calculation. Defaults to 5.0.
        """
        super().__init__(objective)
        self.objective = objective
        self.snr_gamma = snr_gamma

    def forward(self, timesteps, alphas_cumprod):
        """
        Calculates the forward pass of the MinSNRWeighting model.

        Args:
            timesteps (torch.Tensor): A tensor of shape (B,) containing the timesteps.
            alphas_cumprod (torch.Tensor): A tensor of shape (T,) containing the cumulative product of alphas.

        Returns:
            torch.Tensor: A tensor of shape (B,) containing the weighted SNR values.

        Raises:
            NotImplementedError: If the objective is not "noise".
        """
        

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)[timesteps]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)[timesteps]

        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2

        if self.objective == "eps":
            return torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        
        elif self.objective == "x0":
            return torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        else: 
            raise NotImplementedError