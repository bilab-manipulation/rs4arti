import torch as th
from stable_baseline3.sac.policies import Actor, MlpPolicy
from stable_baselines3.common.type_aliases import PyTorchObs

class MyActor(Actor):
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

class MyMlpPolicy(MlpPolicy):
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)