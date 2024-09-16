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
                                                   

class Robot(EnvBase):
    # metadata = {
    #     "render_modes": ["human", "rgb_array"],
    #     "render_fps": 30,
    # }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # Helpers: _make_step and gen_params
        gen_params = staticmethod(gen_params)
        _make_spec = _make_spec

        # Mandatory methods: _step, _reset and _set_seed
        _reset = _reset
        _step = staticmethod(_step)
        _set_seed = _set_seed

    def _step(tensordict):
        # координаты робота
        x_robot = tensordict["xr"]
        y_robot = tensordict["yr"]  
        # координаты цели
        x_target = tensordict["xt"]
        y_target = tensordict["yt"]
        
        angl = tensordict["angl"]
        
        v_action = tensordict["action","v"].squeeze(-1) # линейая скорость робота
        w_action = tensordict["action","w"].squeeze(-1) # угловая скорость робота
        
        dt = tensordict["params", "dt"]
        d_angl_old = tensordict["params","d_angl"]
        
        v_action =  v_action.clamp(-tensordict["params", "max_action"], tensordict["params", "max_action"])
        w_action =  w_action.clamp(-tensordict["params", "max_action"], tensordict["params", "max_action"])
        # reward
        
        # координаты робота
        old_x_robot = tensordict["params","xr"]
        old_y_robot = tensordict["params","yr"]  
        # координаты цели
        old_x_target = tensordict["params","xt"]
        old_y_target = tensordict["params","yt"]
        
        
        new_angl = angl + (w_action * dt) 
        new_x_robot = x_robot + v_action * np.cos(new_angl) * dt
        new_y_robot = y_robot + v_action * np.sin(new_angl) * dt
        
        if( - AREA_DEFEAT > x_robot or x_robot > AREA_DEFEAT  or
            - AREA_DEFEAT > y_robot or y_robot > AREA_DEFEAT):
            reward_target = -REWARD
            done = True
            
        elif ( x_target + AREA_WIN > x_robot > x_target - AREA_WIN   and
                y_target + AREA_WIN > y_robot > y_target - AREA_WIN):
            done = True
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
            
            if dist_new < dist_old:
                dist_reward = 17
            else:
                dist_reward = 0
            # angle
            delta_angle = d_ang(x_robot, y_robot,
                                    x_target, y_target,
                                    new_angl
                                    ).get_angle_dev()

            if delta_angle == 0.0:
                angle_reward = 17
            elif abs(delta_angle) < np.pi / 2 and abs(delta_angle) < abs(d_angl_old):
                angle_reward = -10.85 * delta_angle + 9.28
            else:
                angle_reward = 0
            
        if abs(reward_target) != REWARD:    
            costs = dist_reward + angle_reward
        else:
            costs = reward_target   
            
        reward = -costs.view(*tensordict.shape, 1)    
        done = torch.zeros_like(reward, dtype=torch.bool)
        out = TensorDict(
            {
                "xr": new_x_robot, 
                "yr": new_y_robot,
                "xt":  x_target,
                "yt":  y_target,
                "angl":  new_angl,
                
                ["params","d_angl"]:delta_angle,
                ["params","xt"]:x_target,
                ["params","yt"]:y_target,
                
                "params": tensordict["params"],
                
                "reward": reward,
                
                "done": done,
            },
            tensordict.shape,
        )
        return out

    def _reset(self, tensordict):
        if tensordict is None or tensordict.is_empty():
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
            {   "xr": 0, 
                "yr": 0,
                "xt": xt,
                "yt": yt,
                "angl": 0,
                "params": tensordict["params"],
            },
            batch_size=tensordict.shape,
        )
        return out
    
    def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
        self.observation_spec = Composite(
            xr=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=(1,),dtype=torch.float32),
            yr=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=(1,),dtype=torch.float32),
            xt=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=(1,),dtype=torch.float32),
            yt=Bounded(low=-CORD_RANGE,high=CORD_RANGE,shape=(1,),dtype=torch.float32),
            angl = Bounded(low=-ANGL_RANGE,high=ANGL_RANGE,shape=(1,),dtype=torch.float32),
            
            v_action = Bounded(low=-td_params["params", "max_action"],high=td_params["params", "max_action"],shape=(),dtype=torch.float32), # линейая скорость робота
            w_action = Bounded(low=-td_params["params", "max_action"],high=td_params["params", "max_action"],shape=(),dtype=torch.float32), # угловая скорость робота
           
            params=self.make_composite_from_td(td_params["params"]),
            shape=(),
        )
        
        self.state_spec = self.observation_spec.clone()
        
        self.action_spec = Bounded(low=-td_params["params", "max_action"],high=td_params["params", "max_action"],shape=(2,),dtype=torch.float32)
        
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

    def gen_params(batch_size=None) -> TensorDictBase:
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_action": 0.5,
                        "dt": 0.1,
                        "d_angl": 0,
                        "xr": 0,
                        "yr": 0,
                        "xt": 0,
                        "yt": 0,
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

env = Robot()
check_env_specs(env)
print("observation_spec:", env.observation_spec)
print("state_spec:", env.state_spec)
print("reward_spec:", env.reward_spec)
td = env.reset()
print("reset tensordict", td)