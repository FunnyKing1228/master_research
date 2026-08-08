import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import math


class EvidentialHead(nn.Module):
    """
    Evidential Uncertainty Head
    
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        activation: str = 'relu',
        dropout: float = 0.1,
        gamma_regularizer: float = 1e-2,
        epsilon: float = 1e-6
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gamma_regularizer = gamma_regularizer
        self.epsilon = epsilon
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout),
            self.activation
        )
        
        self.mu_head = nn.Linear(hidden_dim // 2, output_dim)
        self.log_lambda_head = nn.Linear(hidden_dim // 2, output_dim)  # log(λ)
        self.log_alpha_head = nn.Linear(hidden_dim // 2, output_dim)  # log(α-1)
        self.log_beta_head = nn.Linear(hidden_dim // 2, output_dim)  # log(β)
        
        self._init_weights()
        
        print(f"✓ EvidentialHead initialized:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Output dimension: {output_dim}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Gamma regularizer: {gamma_regularizer}")
        print(f"  - Epsilon: {epsilon}")
    
    def _init_weights(self):
        """Documentation for this public API is provided in English."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        
        Args:
            
        Returns:
        """
        features = self.feature_extractor(x)
        
        mu = self.mu_head(features)
        log_lambda = self.log_lambda_head(features)  # log(λ)
        log_alpha_minus_1 = self.log_alpha_head(features)  # log(α-1)
        log_beta = self.log_beta_head(features)  # log(β)
        
        lambda_val = F.softplus(log_lambda) + self.epsilon
        alpha = F.softplus(log_alpha_minus_1) + 1.0 + self.epsilon  # α > 1
        beta = F.softplus(log_beta) + self.epsilon
        
        return mu, lambda_val, alpha, beta
    
    def loss(self, y: torch.Tensor, mu: torch.Tensor, lambda_val: torch.Tensor, 
             alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        
        Args:
            
        Returns:
        """
        nll_loss = self._nll_loss(y, mu, lambda_val, alpha, beta)
        
        reg_loss = self._regularization_loss(y, mu, lambda_val, alpha)
        
        total_loss = nll_loss + self.gamma_regularizer * reg_loss
        
        loss_components = {
            'nll_loss': nll_loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _nll_loss(self, y: torch.Tensor, mu: torch.Tensor, lambda_val: torch.Tensor, 
                   alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        """
        
        lambda_term = lambda_val * (y - mu) ** 2
        beta_term = 2 * beta + lambda_term
        
        log_term = torch.log(lambda_val / (2 * math.pi))
        alpha_term = alpha * torch.log(beta_term / (2 * beta))
        beta_term_log = torch.log(beta_term)
        
        nll = 0.5 * (log_term - alpha_term + beta_term_log)
        
        return nll.mean()
    
    def _regularization_loss(self, y: torch.Tensor, mu: torch.Tensor, 
                            lambda_val: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        """
        prediction_error = torch.abs(y - mu)
        evidence_penalty = prediction_error * (2 * lambda_val + alpha)
        
        return evidence_penalty.mean()
    
    def get_uncertainty(self, mu: torch.Tensor, lambda_val: torch.Tensor, 
                        alpha: torch.Tensor, beta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Args:
            
        Returns:
        """
        aleatoric = beta / (alpha - 1)
        
        epistemic = beta / (lambda_val * (alpha - 1))
        
        return aleatoric, epistemic
    
    def get_total_uncertainty(self, mu: torch.Tensor, lambda_val: torch.Tensor, 
                              alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            
        Returns:
        """
        aleatoric, epistemic = self.get_uncertainty(mu, lambda_val, alpha, beta)
        total_uncertainty = torch.sqrt(aleatoric + epistemic)
        return total_uncertainty
    
    def sample_predictions(self, mu: torch.Tensor, lambda_val: torch.Tensor, 
                           alpha: torch.Tensor, beta: torch.Tensor, 
                           num_samples: int = 100) -> torch.Tensor:
        """
        
        Args:
            
        Returns:
        """
        batch_size = mu.shape[0]
        output_dim = mu.shape[1]
        
        sigma_squared = torch.distributions.InverseGamma(alpha, beta).sample([num_samples])
        sigma_squared = sigma_squared.transpose(0, 1)  # [batch_size, num_samples, output_dim]
        
        sigma = torch.sqrt(sigma_squared)
        samples = torch.distributions.Normal(mu.unsqueeze(1), sigma / torch.sqrt(lambda_val.unsqueeze(1))).sample()
        
        return samples.transpose(0, 1)  # [num_samples, batch_size, output_dim]


class EvidentialCritic(nn.Module):
    """
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        evidential: bool = True,
        gamma_regularizer: float = 1e-2
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.evidential = evidential
        self.gamma_regularizer = gamma_regularizer
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if evidential:
            self.evidential_head = EvidentialHead(
                input_dim=hidden_dim,
                output_dim=1,
                hidden_dim=hidden_dim // 2,
                gamma_regularizer=gamma_regularizer
            )
        else:
            self.q_head = nn.Linear(hidden_dim, 1)
        
        print(f"✓ EvidentialCritic initialized:")
        print(f"  - State dimension: {state_dim}")
        print(f"  - Action dimension: {action_dim}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Evidential: {evidential}")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            
        Returns:
        """
        x = torch.cat([state, action], dim=-1)
        
        features = self.feature_net(x)
        
        if self.evidential:
            mu, lambda_val, alpha, beta = self.evidential_head(features)
            return mu, lambda_val, alpha, beta
        else:
            q_value = self.q_head(features)
            return q_value
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        """
        if self.evidential:
            mu, _, _, _ = self.forward(state, action)
            return mu
        else:
            return self.forward(state, action)
    
    def get_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        """
        if not self.evidential:
            return torch.zeros(state.shape[0], 1, device=state.device)
        
        mu, lambda_val, alpha, beta = self.forward(state, action)
        return self.evidential_head.get_total_uncertainty(mu, lambda_val, alpha, beta)


def create_evidential_critic(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    evidential: bool = True,
    gamma_regularizer: float = 1e-2
) -> EvidentialCritic:
    """
    
    Args:
        
    Returns:
    """
    return EvidentialCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        evidential=evidential,
        gamma_regularizer=gamma_regularizer
    ) 