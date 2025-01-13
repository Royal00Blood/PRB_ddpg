from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import  CompositeSpec as Composite
from torchrl.data import  UnboundedContinuousTensorSpec as Unbounded
from torchrl.data import BoundedTensorSpec as Bounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from settings import AREA_GENERATION, REWARD, AREA_DEFEAT, AREA_WIN
from calculation_scripts.deviation_angle import DeviationAngle as d_ang
from calculation_scripts.DistanceBW2points import DistanceBW2points as d_dist

CORD_RANGE = 1000.00
ANGL_RANGE = 3.14
                                                   

class RobotEnv(EnvBase):
    
    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        
        # self.old_x_robot  = torch.zeros(1)
        # self.old_y_robot  = torch.zeros(1)
        # self.old_x_target = torch.zeros(1) 
        # self.old_y_target = torch.zeros(1)

    def _step(self, tensordict):
        # даные состояния среды
        # координаты робота
        x_robot = tensordict["xr"]
        y_robot = tensordict["yr"]  
        # координаты цели
        x_target = tensordict["xt"]
        y_target = tensordict["yt"]
        # угол робота
        angle = tensordict["angle"]
        #
        old_x_robot  = tensordict["old_xr"]
        old_y_robot  = tensordict["old_yr"]
        old_x_target = tensordict["old_xt"]
        old_y_target = tensordict["old_yt"]

        action = tensordict["action"].squeeze(-1) # линейая скорость робота
        # Котанты
        dt = tensordict["params", "dt"]
        action = action.clamp(-tensordict["params", "max_a"], tensordict["params", "max_a"])
        
        # reward
        reward_target = 0
        
        new_angle = angle + (action[1] * dt) 
        
        new_x_robot = x_robot + action[0] * np.cos(new_angle) * dt
        new_y_robot = y_robot + action[0] * np.sin(new_angle) * dt
        # print(f"xr = {new_x_robot.size()}, yr = {new_y_robot.size()},angle = {new_angle.size()} ")
        if( - AREA_DEFEAT > x_robot or x_robot > AREA_DEFEAT  or
            - AREA_DEFEAT > y_robot or y_robot > AREA_DEFEAT):
            done = torch.tensor(True)
            reward_target = -REWARD
        elif ( x_target + AREA_WIN > x_robot > x_target - AREA_WIN   and
                y_target + AREA_WIN > y_robot > y_target - AREA_WIN):
            done = torch.tensor(True)
            reward_target = REWARD
        else:
            # dist
            # координаты цели
            if old_x_target==0:
                old_x_target = x_target
                old_y_target = y_target
            dist_new = d_dist(x_target, y_target, x_robot, y_robot).getDistance()
            dist_old = d_dist(old_x_target, old_y_target, old_x_robot, old_y_robot).getDistance()
            # координаты робота
            old_x_robot = x_robot
            old_y_robot = y_robot  
            # координаты цели
            old_x_target = x_target
            old_y_target = y_target
            # награда за изменение растояния
            dist_reward = dist_old - dist_new
            # angle
            delta_angle = d_ang(x_robot.numpy(), y_robot.numpy(), x_target.numpy(), y_target.numpy(), new_angle.numpy()).get_angle_dev()
            delta_angle = torch.tensor(delta_angle)
            angle_reward = -abs(delta_angle)
            reward_target = dist_reward + angle_reward
            reward = torch.tensor(reward_target).view(*tensordict.shape, 1)
            done = torch.tensor(False)
            
        # reward = reward_target.view(*tensordict.shape, 1)    
        out = TensorDict(
            {
                "xr": torch.tensor(new_x_robot), 
                "yr": torch.tensor(new_y_robot),
                "xt": torch.tensor(x_target),
                "yt": torch.tensor(y_target),
                "angle": new_angle,
                "old_xr": old_x_robot,
                "old_yr": old_y_robot,
                "old_xt": old_x_target,
                "old_yt": old_y_target,
                "params": tensordict["params"],
                "reward": reward,
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        print(f"tensordict: {tensordict}")
        if (tensordict is None) or tensordict.is_empty():
            tensordict = self.gen_params(batch_size=self.batch_size)

        
        
        high_xt = torch.tensor(AREA_GENERATION, device=self.device)
        high_yt = torch.tensor(AREA_GENERATION, device=self.device)
        low_xt = -high_xt
        low_yt = -high_yt

        xt = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_xt - low_xt)
            + low_xt
        )
        yt = (
            torch.rand(tensordict.shape, generator=self.rng, device=self.device)
            * (high_yt - low_yt)
            + low_yt
        )
        out = TensorDict(
            {   "xr":    torch.zeros(tensordict.shape, device=self.device), 
                "yr":    torch.zeros(tensordict.shape, device=self.device),
                "xt":    xt,
                "yt":    yt,
                "angle": torch.zeros(tensordict.shape, device=self.device),
                "old_xr": torch.zeros(tensordict.shape, device=self.device),
                "old_yr": torch.zeros(tensordict.shape, device=self.device),
                "old_xt": torch.zeros(tensordict.shape, device=self.device),
                "old_yt": torch.zeros(tensordict.shape, device=self.device),
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out
    
    def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            xr=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            yr=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            xt=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            yt=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            angle = Bounded(low=-ANGL_RANGE,high=ANGL_RANGE,shape=()),
            old_xr=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            old_yr=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            old_xt=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()),
            old_yt=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=()), 
            params=self.make_composite_from_td(td_params["params"]),
            shape=(),
        )
        
        self.state_spec = self.observation_spec.clone()
        
        self.action_spec = Bounded(low=-td_params["params", "max_a"],high=td_params["params", "max_a"],shape=(2,),dtype=torch.float32) ######
        
        self.reward_spec = Unbounded(shape=(*td_params.shape, 1))
    
    def make_composite_from_td(self,td):
        # custom function to convert a ``tensordict`` in a similar spec structure
        # of unbounded values.
        composite = Composite(
            {
                key: self.make_composite_from_td(tensor)
                if isinstance(tensor, TensorDictBase)
                else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
                for key, tensor in td.items()
            },
            shape=td.shape,
        )
        return composite      

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    @staticmethod
    def gen_params( batch_size=None) -> TensorDictBase:
        if batch_size is None:
            batch_size = []
        
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_a": torch.tensor(0.5),
                        "dt": torch.tensor(0.1),
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

env = RobotEnv()
# env = TransformedEnv(
#     env,
#     UnsqueezeTransform(
#         dim=-1,
#         in_keys=["xr", "yr","xt","yt","angle","old_xr","old_yr","old_xt","old_yt"],
#         in_keys_inv=["xr", "yr","xt","yt","angle","old_xr","old_yr","old_xt","old_yt"],
#     ),
# )

check_env_specs(env)

# def simple_rollout(steps=100):
#     # preallocate:
#     data = TensorDict({}, [steps])
#     # reset
#     _data = env.reset()
#     for i in range(steps):
#         _data["action"] = env.action_spec.rand()
#         _data = env.step(_data)
#         data[i] = _data
#         _data = step_mdp(_data, keep_other=True)
#     return data


# print("data from rollout:", simple_rollout(100))



batch_size = 10 
gen = env.gen_params(batch_size=batch_size)
#print(f"gen: {gen.shape}, {gen.type}, {gen}")# number of environments to be executed in batch
td = env.reset(env.gen_params(batch_size=batch_size))
#print("reset (batch size of 10)", td)
td = env.rand_step(td)
#print("rand step (batch size of 10)", td)

# rollout = env.rollout(
#     3,
#     auto_reset=False,  # we're executing the reset out of the ``rollout`` call
#     tensordict=env.reset(env.gen_params(batch_size=[batch_size])),
# )
# print("rollout of len 3 (batch size of 10):", rollout)
                 