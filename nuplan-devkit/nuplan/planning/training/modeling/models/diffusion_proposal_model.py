from typing import cast

import torch
import torch.nn as nn
import numpy as np

from diffusers import DDIMScheduler

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import (
    convert_predictions_to_trajectory,
)
from nuplan.planning.training.modeling.models.diffusion_utils import (
    Standardizer,
    SinusoidalPosEmb,
    VerletStandardizer
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.models.encoder_decoder_layers import (
    ParallelAttentionLayer
)
from nuplan.planning.training.preprocessing.features.agent_history import AgentHistory
from nuplan.planning.training.preprocessing.feature_builders.agent_history_feature_builder import (
    AgentHistoryFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.agent_trajectory_target_builder import (
    AgentTrajectoryTargetBuilder
)
from nuplan.planning.training.modeling.models.positional_embeddings import RotaryPositionEncoding


class DiffusionProposalModel(TorchModuleWrapper):
    def __init__(
        self,
        feature_dim,
        past_trajectory_sampling,
        future_trajectory_sampling,
        map_params,
        T: int = 32,
        predictions_per_sample: int = 16, # Number of trajectories predicted by a sample
        max_dist: float = 200,           # used to normalize ALL tokens (incl. trajectory)
        easy_validation: bool = False,  # instead of starting from pure noise, start with not that much noise at inference,
        use_verlet: bool = True,
        ignore_history: bool = False
    ):
        super().__init__(
            feature_builders=[
                AgentHistoryFeatureBuilder(
                    trajectory_sampling=past_trajectory_sampling,
                    max_agents=10
                ),
                VectorSetMapFeatureBuilder(
                    map_features=map_params['map_features'],
                    max_elements=map_params['max_elements'],
                    max_points=map_params['max_points'],
                    radius=map_params['vector_set_map_feature_radius'],
                    interpolation_method=map_params['interpolation_method']
                )
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(future_trajectory_sampling),
                AgentTrajectoryTargetBuilder(
                    trajectory_sampling=past_trajectory_sampling,
                    future_trajectory_sampling=future_trajectory_sampling,
                    max_agents=10
                )
            ],
            future_trajectory_sampling=future_trajectory_sampling
        )

        self.feature_dim = feature_dim
        self.T = T
        self.H = 16 # Trajectory time dimension size
        self.output_dim = self.H * 3 #  x (x,y,theta) is the dimension size of a traj
        self.predictions_per_sample = predictions_per_sample # Number of trajectories predicted by a sample
        self.max_dist = max_dist
        self.easy_validation = easy_validation
        self.use_verlet = use_verlet
        self.ignore_history = ignore_history

        self.standardizer = VerletStandardizer() if use_verlet else Standardizer(max_dist=max_dist)

        # DIFFUSER
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.T,
            beta_schedule='scaled_linear',
            prediction_type='epsilon',
        )
        self.scheduler.set_timesteps(self.T)

        self.history_encoder = nn.Sequential(
            nn.Linear(7, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.d_history_encoder = nn.Sequential(
            nn.Linear(7, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.dd_history_encoder = nn.Sequential(
            nn.Linear(7, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        self.sigma_encoder = nn.Sequential(
            SinusoidalPosEmb(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )
        self.sigma_proj_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

        self.trajectory_encoder = nn.Linear(3, self.feature_dim)
        self.d_trajectory_encoder = nn.Linear(3, self.feature_dim)
        self.dd_trajectory_encoder = nn.Linear(3, self.feature_dim)

        self.pva_encoder = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )


        self.trajectory_time_embeddings = RotaryPositionEncoding(self.feature_dim)

        # Category embedding: The model can automatically adjust these embedding vectors according to the input categories to improve the performance of the model.
        # 0: Scene
        # 1: traj
        # 2: Noise
        # self.type_embedding = nn.Embedding(3, self.feature_dim) # trajectory, noise token

        self.global_attention_layers = torch.nn.ModuleList([
            ParallelAttentionLayer(
                d_model=self.feature_dim, 
                self_attention1=True, self_attention2=False,
                cross_attention1=False, cross_attention2=False,
                rotary_pe=True
            )
        for _ in range(8)])
        
        # decoder, multi-layer perceptron, feature to three-dimensional traj
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 3)
        )

        self.all_trajs = []
        self.apply(self._init_weights)
        self.precompute_variances()

    # Initialize the weight of the module
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Pre-calculate variance
    def precompute_variances(self):
        """
        Precompute variances from alphas
        """
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[self.scheduler.timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[self.scheduler.timesteps]) ** 0.5
        self._variances = sqrt_one_minus_alpha_prod / sqrt_alpha_prod

    
    def sigma(self, t):
        return t

    def forward(self, features: FeaturesType, num_grad_steps=15, step_size=.001, use_clean=False) -> TargetsType:
        # Recover features
        ego_agent_features = cast(AgentHistory, features["agent_history"]) # including traj his, agents his、mask available time step
        batch_size = ego_agent_features.batch_size

        state_features = self.encode_scene_features(ego_agent_features) # ego his feature, scene type embedding

        # Only use for denoising
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone() # B x H x D
            # Normalization x,y,theta to [0,1]
            ego_gt_trajectory = self.standardizer.transform_features(ego_agent_features, ego_gt_trajectory)

        noise = torch.randn(ego_gt_trajectory.shape, device=ego_gt_trajectory.device)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (ego_gt_trajectory.shape[0],), 
                                    device=ego_gt_trajectory.device).long() 
        ego_noisy_trajectory = self.scheduler.add_noise(ego_gt_trajectory, noise, timesteps)

        pred_noise = self.denoise(ego_noisy_trajectory, timesteps, state_features)

        output = {
            # TODO: fix trajectory, right now we only care about epsilon at training and this is a dummy value
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(pred_noise)), # required for nuplan training
            "epsilon": pred_noise,
            "gt_epsilon": noise
        }
        return output
        
    def encode_scene_features(self, ego_agent_features):
        ego_features = ego_agent_features.ego
        if self.ignore_history:
            ego_features = torch.zeros_like(ego_features)
        d_ego_features = torch.diff(ego_features, dim=1)
        d_ego_features_padded = torch.cat([d_ego_features, d_ego_features[:,-1:,:]], dim=1) # B X 5 X 7
        dd_ego_features = torch.diff(d_ego_features, dim=1)
        dd_ego_features_padded = torch.cat([dd_ego_features[:,:1,:], dd_ego_features, dd_ego_features[:,-1:,:]], dim=1) # B X 5 X 7

        ego_features = self.history_encoder(ego_features) # Bx5x7 -> Bx5xD 
        d_ego_features_padded = self.d_history_encoder(d_ego_features_padded) # Bx5x7 -> Bx5xD 
        dd_ego_features_padded = self.dd_history_encoder(dd_ego_features_padded) # Bx5x7 -> Bx5xD 
        
        ego_features = self.pva_encoder(torch.cat([ego_features,d_ego_features_padded,dd_ego_features_padded], dim=2))

        # ego_type_embedding = self.type_embedding(torch.as_tensor([[0]], device=ego_features.device))
        # ego_type_embedding = ego_type_embedding.repeat(ego_features.shape[0],5,1)

        return ego_features

    def denoise(self, ego_trajectory, sigma, state_features):
        batch_size = ego_trajectory.shape[0]

        state_features = state_features
        
        # Trajectory features
        ego_trajectory = ego_trajectory.reshape(ego_trajectory.shape[0],self.H,3) # B X H X 3
        d_traj = torch.diff(ego_trajectory, dim=1) # B X H-1 X 3
        d_traj_padded = torch.cat([d_traj, d_traj[:,-1:,:]], dim=1) # B X H X 3
        dd_traj = torch.diff(d_traj, dim=1) # B X H-2 X 3
        dd_traj_padded = torch.cat([dd_traj[:, :1, :], dd_traj, dd_traj[:,-1:,:]], dim=1) # B X H X 3

        trajectory_features = self.trajectory_encoder(ego_trajectory)# B X H X D
        d_trajectory_features = self.d_trajectory_encoder(d_traj_padded)# B X H X D
        dd_trajectory_features = self.dd_trajectory_encoder(dd_traj_padded)# B X H X D

        trajectory_features = self.pva_encoder(torch.cat([trajectory_features,d_trajectory_features,dd_trajectory_features],dim=2)) # B X H X 3D -> B X H X D

        # trajectory_type_embedding = self.type_embedding(
        #     torch.as_tensor([1], device=ego_trajectory.device)
        # )[None].repeat(batch_size,self.H,1)
        
        # Concatenate all features
        all_features = torch.cat([state_features, trajectory_features], dim=1)
        # all_type_embedding = torch.cat([state_type_embedding, trajectory_type_embedding], dim=1)

        # Sigma encoding
        sigma = sigma.reshape(-1,1)
        # .numel() is an operation used to get the total number of all elements in a tensor.
        if sigma.numel() == 1:
            sigma = sigma.repeat(batch_size,1)
        sigma = sigma.float() / self.T
        sigma_embeddings = self.sigma_encoder(sigma)

        # Increase time dimension
        sigma_embeddings = sigma_embeddings.reshape(batch_size,1,self.feature_dim)

        # Concatenate sigma features and project back to original feature_dim
        sigma_embeddings = sigma_embeddings.repeat(1,all_features.shape[1],1)
        all_features = torch.cat([all_features, sigma_embeddings], dim=2)
        # all_features retains the original feature dimension size and combines the information of sigma.
        all_features = self.sigma_proj_layer(all_features)

        # Generate attention mask
        seq_len = all_features.shape[1]
        indices = torch.arange(seq_len, device=all_features.device)
        # indices[None] seq_len, -> 1,seq_len
        # indices[:,None] seq_len, -> seq_len,1
        dists = (indices[None] - indices[:,None]).abs()
        attn_mask = dists > 1       # TODO: magic number

        # Generate relative temporal embeddings
        temporal_embedding = self.trajectory_time_embeddings(indices[None].repeat(batch_size,1))

        # Global self-attentions
        for layer in self.global_attention_layers:            
            all_features, _ = layer(
                all_features, None, None, None,
                seq1_pos=temporal_embedding,
                seq1_sem_pos=None,
                attn_mask_11=attn_mask
            )

        trajectory_features = all_features[:,-self.H:]
        out = self.decoder_mlp(trajectory_features).reshape(trajectory_features.shape[0],-1)

        return out # , all_weights

    def run_diffusion(
        self, 
        features: FeaturesType
    ) -> TargetsType:
        # Recover features
        ego_agent_features = cast(AgentHistory, features["agent_history"])
        batch_size = ego_agent_features.batch_size * self.predictions_per_sample

        state_features = self.encode_scene_features(ego_agent_features) 

        # Only use for denoising 
        if 'trajectory' in features:
            ego_gt_trajectory = features['trajectory'].data.clone()
            ego_gt_trajectory = self.standardizer.transform_features(ego_agent_features, ego_gt_trajectory)
        # print("state_features",state_features.shape)
        # Multiple predictions per sample
        state_features = (
            state_features.repeat_interleave(self.predictions_per_sample,0)
        )
        # print("state_features",state_features.shape)
        ego_agent_features = ego_agent_features.repeat_interleave(self.predictions_per_sample,0)


        trajectory_shape = (batch_size, self.H * 3)
        device = state_features[0].device


        # Initialize elite set with random 
        noise = torch.randn(trajectory_shape, device=device)
        population_trajectories = self.rollout(
            features,
            state_features,
            noise,
            deterministic=False,
        )

        out = {
            "multimodal_trajectories": population_trajectories,
        }

        return out
    
    def rollout(
        self,
        features,
        state_features,
        ego_trajectory,
        deterministic=True, 
        noise_scale=1.0, 
    ):

        timesteps = self.scheduler.timesteps

        for t in timesteps:
            residual = torch.zeros_like(ego_trajectory)

            with torch.no_grad():
                # Noise predicted at each step
                residual += self.denoise(ego_trajectory, t.to(ego_trajectory.device), state_features)

            if deterministic:
                eta = 0.0 # The coefficient of noise added by the reverse process
            else:
                prev_alpha = self.scheduler.alphas[t-1]
                alpha = self.scheduler.alphas[t]
                eta = noise_scale * torch.sqrt((1 - prev_alpha) / (1 - alpha)) * \
                        torch.sqrt((1 - alpha) / prev_alpha)

            out = self.scheduler.step(residual, t, ego_trajectory, eta=eta)
            ego_trajectory = out.prev_sample

        ego_agent_features = cast(AgentHistory, features["agent_history"])
        ego_trajectory = self.standardizer.untransform_features(ego_agent_features, ego_trajectory)

        return ego_trajectory

    def renoise(self, ego_trajectory, t):
        noise = torch.randn(ego_trajectory.shape, device=ego_trajectory.device)
        ego_trajectory = self.scheduler.add_noise(ego_trajectory, noise, self.scheduler.timesteps[-t])
        return ego_trajectory


def compute_constraint_scores(constraints, trajectory):
    all_info = {}
    total_cost = torch.zeros(trajectory.shape[0], device=trajectory.device)
    # constraints Functions can have multiple
    for constraint in constraints:
        cost, info = constraint(trajectory) # Call constraint fn
        total_cost += cost
        all_info.update(info)
    return total_cost, all_info


def compute_constraint_residual(constraints, trajectory):
    """
    Compute the gradient of the sum of all the constraints w.r.t trajectory
    """
    with torch.enable_grad():
        trajectory.requires_grad_(True)
        total_cost, _ = compute_constraint_scores(constraints, trajectory)
        total_cost.mean().backward() # Backpropagation
        grad = trajectory.grad
        trajectory.requires_grad_(False)
    return grad
