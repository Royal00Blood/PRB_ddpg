import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
try:
    multiprocessing.set_start_method("spawn" if is_sphinx else "fork")
except RuntimeError:
    pass
from torchrl.objectives.utils import ValueEstimators
default_value_estimator = ValueEstimators.TD0
from torchrl.objectives.utils import default_value_kwargs
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator
import torch
import tqdm
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
collector_device = torch.device("cpu")
from torchrl.objectives.utils import distance_loss
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ActorCriticWrapper
from tensordict import TensorDict, TensorDictBase
from torchrl.objectives import LossModule
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv

env_library = None
env_name = None

from torchrl.envs import (
    CatTensors,
    DoubleToFloat,
    EnvCreator,
    InitTracker,
    ObservationNorm,
    ParallelEnv,
    RewardScaling,
    StepCounter,
    TransformedEnv,
)


def _init(self,actor_network: TensorDictModule,value_network: TensorDictModule,) -> None:
    super(type(self), self).__init__()
    
    self.convert_to_functional(
        actor_network,
        "actor_network",
        create_target_params=True,
    )
    self.convert_to_functional(
        value_network,
        "value_network",
        create_target_params=True,
        compare_against=list(actor_network.parameters()),
    )
    
    self.actor_in_keys = actor_network.in_keys
    
    actor_critic = ActorCriticWrapper(actor_network, value_network)
    self.actor_critic = actor_critic
    self.loss_function = "l2"
    
def make_value_estimator(self, value_type: ValueEstimators, **hyperparams):
    hp = dict(default_value_kwargs(value_type))
    if hasattr(self, "gamma"):
        hp["gamma"] = self.gamma
    hp.update(hyperparams)
    value_key = "state_action_value"
    if value_type == ValueEstimators.TD1:
        self._value_estimator = TD1Estimator(value_network=self.actor_critic, **hp)
    elif value_type == ValueEstimators.TD0:
        self._value_estimator = TD0Estimator(value_network=self.actor_critic, **hp)
    elif value_type == ValueEstimators.GAE:
        raise NotImplementedError(
            f"Value type {value_type} it not implemented for loss {type(self)}."
        )
    elif value_type == ValueEstimators.TDLambda:
        self._value_estimator = TDLambdaEstimator(value_network=self.actor_critic, **hp)
    else:
        raise NotImplementedError(f"Unknown value type {value_type}")
    self._value_estimator.set_keys(value=value_key)

def _loss_actor(
    self,
    tensordict,
) -> torch.Tensor:
    td_copy = tensordict.select(*self.actor_in_keys)
    # Get an action from the actor network: since we made it functional, we need to pass the params
    with self.actor_network_params.to_module(self.actor_network):
        td_copy = self.actor_network(td_copy)
    # get the value associated with that action
    with self.value_network_params.detach().to_module(self.value_network):
        td_copy = self.value_network(td_copy)
    return -td_copy.get("state_action_value")

def _loss_value(
    self,
    tensordict,
):
    td_copy = tensordict.clone()

    # V(s, a)
    with self.value_network_params.to_module(self.value_network):
        self.value_network(td_copy)
    pred_val = td_copy.get("state_action_value").squeeze(-1)

    # we manually reconstruct the parameters of the actor-critic, where the first
    # set of parameters belongs to the actor and the second to the value function.
    target_params = TensorDict(
        {
            "module": {
                "0": self.target_actor_network_params,
                "1": self.target_value_network_params,
            }
        },
        batch_size=self.target_actor_network_params.batch_size,
        device=self.target_actor_network_params.device,
    )
    with target_params.to_module(self.actor_critic):
        target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)

    # Computes the value loss: L2, L1 or smooth L1 depending on `self.loss_function`
    loss_value = distance_loss(pred_val, target_value, loss_function=self.loss_function)
    td_error = (pred_val - target_value).pow(2)

    return loss_value, td_error, pred_val, target_value

def _forward(self, input_tensordict: TensorDictBase) -> TensorDict:
    loss_value, td_error, pred_val, target_value = self.loss_value(
        input_tensordict,
    )
    td_error = td_error.detach()
    td_error = td_error.unsqueeze(input_tensordict.ndimension())
    if input_tensordict.device is not None:
        td_error = td_error.to(input_tensordict.device)
    input_tensordict.set(
        "td_error",
        td_error,
        inplace=True,
    )
    loss_actor = self.loss_actor(input_tensordict)
    return TensorDict(
        source={
            "loss_actor": loss_actor.mean(),
            "loss_value": loss_value.mean(),
            "pred_value": pred_val.mean().detach(),
            "target_value": target_value.mean().detach(),
            "pred_value_max": pred_val.max().detach(),
            "target_value_max": target_value.max().detach(),
        },
        batch_size=[],
    )

class DDPGLoss(LossModule):
    default_value_estimator = default_value_estimator
    make_value_estimator = make_value_estimator

    __init__ = _init
    forward = _forward
    loss_value = _loss_value
    loss_actor = _loss_actor

def make_env(from_pixels=False):
    """Create a base ``env``."""
    global env_library
    global env_name

    if backend == "dm_control":
        env_name = "cheetah"
        env_task = "run"
        env_args = (env_name, env_task)
        env_library = DMControlEnv
    elif backend == "gym":
        env_name = "HalfCheetah-v4"
        env_args = (env_name,)
        env_library = GymEnv
    else:
        raise NotImplementedError

    env_kwargs = {
        "device": device,
        "from_pixels": from_pixels,
        "pixels_only": from_pixels,
        "frame_skip": 2,
    }
    env = env_library(*env_args, **env_kwargs)
    return env

reward_scaling = 5.0
max_frames_per_traj = 500

def make_transformed_env(
    env,
):
    """Apply transforms to the ``env`` (such as reward scaling and state normalization)."""

    env = TransformedEnv(env)

    # we append transforms one by one, although we might as well create the
    # transformed environment using the `env = TransformedEnv(base_env, transforms)`
    # syntax.
    env.append_transform(RewardScaling(loc=0.0, scale=reward_scaling))

    # We concatenate all states into a single "observation_vector"
    # even if there is a single tensor, it'll be renamed in "observation_vector".
    # This facilitates the downstream operations as we know the name of the
    # output tensor.
    # In some environments (not half-cheetah), there may be more than one
    # observation vector: in this case this code snippet will concatenate them
    # all.
    selected_keys = list(env.observation_spec.keys())
    out_key = "observation_vector"
    env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

    # we normalize the states, but for now let's just instantiate a stateless
    # version of the transform
    env.append_transform(ObservationNorm(in_keys=[out_key], standard_normal=True))

    env.append_transform(DoubleToFloat())

    env.append_transform(StepCounter(max_frames_per_traj))

    # We need a marker for the start of trajectories for our Ornstein-Uhlenbeck (OU)
    # exploration:
    env.append_transform(InitTracker())

    return env




def parallel_env_constructor(
    env_per_collector,
    transform_state_dict,
):
    if env_per_collector == 1:

        def make_t_env():
            env = make_transformed_env(make_env())
            env.transform[2].init_stats(3)
            env.transform[2].loc.copy_(transform_state_dict["loc"])
            env.transform[2].scale.copy_(transform_state_dict["scale"])
            return env

        env_creator = EnvCreator(make_t_env)
        return env_creator

    parallel_env = ParallelEnv(
        num_workers=env_per_collector,
        create_env_fn=EnvCreator(lambda: make_env()),
        create_env_kwargs=None,
        pin_memory=False,
    )
    env = make_transformed_env(parallel_env)
    # we call `init_stats` for a limited number of steps, just to instantiate
    # the lazy buffers.
    env.transform[2].init_stats(3, cat_dim=1, reduce_dim=[0, 1])
    env.transform[2].load_state_dict(transform_state_dict)
    return env


# The backend can be ``gym`` or ``dm_control``
backend = "gym"

def get_env_stats():
    """Gets the stats of an environment."""
    proof_env = make_transformed_env(make_env())
    t = proof_env.transform[2]
    t.init_stats(init_env_steps)
    transform_state_dict = t.state_dict()
    proof_env.close()
    return transform_state_dict


###############################################################################
# Normalization stats
# ~~~~~~~~~~~~~~~~~~~
# Number of random steps used as for stats computation using ``ObservationNorm``

init_env_steps = 5000

transform_state_dict = get_env_stats()

###############################################################################
# Number of environments in each data collector
env_per_collector = 4

###############################################################################
# We pass the stats computed earlier to normalize the output of our
# environment:

parallel_env = parallel_env_constructor(
    env_per_collector=env_per_collector,
    transform_state_dict=transform_state_dict,
)


from torchrl.data import Composite

from torchrl.modules import (
    ActorCriticWrapper,
    DdpgMlpActor,
    DdpgMlpQNet,
    OrnsteinUhlenbeckProcessModule,
    ProbabilisticActor,
    TanhDelta,
    ValueOperator,
)


def make_ddpg_actor(
    transform_state_dict,
    device="cpu",
):
    proof_environment = make_transformed_env(make_env())
    proof_environment.transform[2].init_stats(3)
    proof_environment.transform[2].load_state_dict(transform_state_dict)

    out_features = proof_environment.action_spec.shape[-1]

    actor_net = DdpgMlpActor(
        action_dim=out_features,
    )

    in_keys = ["observation_vector"]
    out_keys = ["param"]

    actor = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=out_keys,
    )

    actor = ProbabilisticActor(
        actor,
        distribution_class=TanhDelta,
        in_keys=["param"],
        spec=Composite(action=proof_environment.action_spec),
    ).to(device)

    q_net = DdpgMlpQNet()

    in_keys = in_keys + ["action"]
    qnet = ValueOperator(
        in_keys=in_keys,
        module=q_net,
    ).to(device)

    # initialize lazy modules
    qnet(actor(proof_environment.reset().to(device)))
    return actor, qnet


actor, qnet = make_ddpg_actor(
    transform_state_dict=transform_state_dict,
    device=device,
)

###############################################################################
# Exploration
# ~~~~~~~~~~~
#
# The policy is passed into a :class:`~torchrl.modules.OrnsteinUhlenbeckProcessModule`
# exploration module, as suggested in the original paper.
# Let's define the number of frames before OU noise reaches its minimum value
annealing_frames = 1_000_000

actor_model_explore = TensorDictSequential(
    actor,
    OrnsteinUhlenbeckProcessModule(
        spec=actor.spec.clone(),
        annealing_num_steps=annealing_frames,
    ).to(device),
)
if device == torch.device("cpu"):
    actor_model_explore.share_memory()


###############################################################################
# Data collector
# --------------
#
# TorchRL provides specialized classes to help you collect data by executing
# the policy in the environment. These "data collectors" iteratively compute
# the action to be executed at a given time, then execute a step in the
# environment and reset it when required.
# Data collectors are designed to help developers have a tight control
# on the number of frames per batch of data, on the (a)sync nature of this
# collection and on the resources allocated to the data collection (for example
# GPU, number of workers, and so on).
#
# Here we will use
# :class:`~torchrl.collectors.SyncDataCollector`, a simple, single-process
# data collector. TorchRL offers other collectors, such as
# :class:`~torchrl.collectors.MultiaSyncDataCollector`, which executed the
# rollouts in an asynchronous manner (for example, data will be collected while
# the policy is being optimized, thereby decoupling the training and
# data collection).
#
# The parameters to specify are:
#
# - an environment factory or an environment,
# - the policy,
# - the total number of frames before the collector is considered empty,
# - the maximum number of frames per trajectory (useful for non-terminating
#   environments, like ``dm_control`` ones).
#
#   .. note::
#
#     The ``max_frames_per_traj`` passed to the collector will have the effect
#     of registering a new :class:`~torchrl.envs.StepCounter` transform
#     with the environment used for inference. We can achieve the same result
#     manually, as we do in this script.
#
# One should also pass:
#
# - the number of frames in each batch collected,
# - the number of random steps executed independently from the policy,
# - the devices used for policy execution
# - the devices used to store data before the data is passed to the main
#   process.
#
# The total frames we will use during training should be around 1M.
total_frames = 10_000  # 1_000_000

###############################################################################
# The number of frames returned by the collector at each iteration of the outer
# loop is equal to the length of each sub-trajectories times the number of
# environments run in parallel in each collector.
#
# In other words, we expect batches from the collector to have a shape
# ``[env_per_collector, traj_len]`` where
# ``traj_len=frames_per_batch/env_per_collector``:
#
traj_len = 200
frames_per_batch = env_per_collector * traj_len
init_random_frames = 5000
num_collectors = 2

from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType

collector = SyncDataCollector(
    parallel_env,
    policy=actor_model_explore,
    total_frames=total_frames,
    frames_per_batch=frames_per_batch,
    init_random_frames=init_random_frames,
    reset_at_each_iter=False,
    split_trajs=False,
    device=collector_device,
    exploration_type=ExplorationType.RANDOM,
)

###############################################################################
# Evaluator: building your recorder object
# ----------------------------------------
#
# As the training data is obtained using some exploration strategy, the true
# performance of our algorithm needs to be assessed in deterministic mode. We
# do this using a dedicated class, ``Recorder``, which executes the policy in
# the environment at a given frequency and returns some statistics obtained
# from these simulations.
#
# The following helper function builds this object:
from torchrl.trainers import Recorder


def make_recorder(actor_model_explore, transform_state_dict, record_interval):
    base_env = make_env()
    environment = make_transformed_env(base_env)
    environment.transform[2].init_stats(
        3
    )  # must be instantiated to load the state dict
    environment.transform[2].load_state_dict(transform_state_dict)

    recorder_obj = Recorder(
        record_frames=1000,
        policy_exploration=actor_model_explore,
        environment=environment,
        exploration_type=ExplorationType.MEAN,
        record_interval=record_interval,
    )
    return recorder_obj


###############################################################################
# We will be recording the performance every 10 batch collected
record_interval = 10

recorder = make_recorder(
    actor_model_explore, transform_state_dict, record_interval=record_interval
)

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    PrioritizedSampler,
    RandomSampler,
    TensorDictReplayBuffer,
)

###############################################################################
# Replay buffer
# -------------
#
# Replay buffers come in two flavors: prioritized (where some error signal
# is used to give a higher likelihood of sampling to some items than others)
# and regular, circular experience replay.
#
# TorchRL replay buffers are composable: one can pick up the storage, sampling
# and writing strategies. It is also possible to
# store tensors on physical memory using a memory-mapped array. The following
# function takes care of creating the replay buffer with the desired
# hyperparameters:
#

from torchrl.envs import RandomCropTensorDict


def make_replay_buffer(buffer_size, batch_size, random_crop_len, prefetch=3, prb=False):
    if prb:
        sampler = PrioritizedSampler(
            max_capacity=buffer_size,
            alpha=0.7,
            beta=0.5,
        )
    else:
        sampler = RandomSampler()
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            buffer_size,
            scratch_dir=buffer_scratch_dir,
        ),
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=False,
        prefetch=prefetch,
        transform=RandomCropTensorDict(random_crop_len, sample_dim=1),
    )
    return replay_buffer


###############################################################################
# We'll store the replay buffer in a temporary directory on disk

import tempfile

tmpdir = tempfile.TemporaryDirectory()
buffer_scratch_dir = tmpdir.name

###############################################################################
# Replay buffer storage and batch size
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# TorchRL replay buffer counts the number of elements along the first dimension.
# Since we'll be feeding trajectories to our buffer, we need to adapt the buffer
# size by dividing it by the length of the sub-trajectories yielded by our
# data collector.
# Regarding the batch-size, our sampling strategy will consist in sampling
# trajectories of length ``traj_len=200`` before selecting sub-trajectories
# or length ``random_crop_len=25`` on which the loss will be computed.
# This strategy balances the choice of storing whole trajectories of a certain
# length with the need for providing samples with a sufficient heterogeneity
# to our loss. The following figure shows the dataflow from a collector
# that gets 8 frames in each batch with 2 environments run in parallel,
# feeds them to a replay buffer that contains 1000 trajectories and
# samples sub-trajectories of 2 time steps each.
#
# .. figure:: /_static/img/replaybuffer_traj.png
#    :alt: Storing trajectories in the replay buffer
#
# Let's start with the number of frames stored in the buffer


def ceil_div(x, y):
    return -x // (-y)


buffer_size = 1_000_000
buffer_size = ceil_div(buffer_size, traj_len)

###############################################################################
# Prioritized replay buffer is disabled by default
prb = False

###############################################################################
# We also need to define how many updates we'll be doing per batch of data
# collected. This is known as the update-to-data or ``UTD`` ratio:
update_to_data = 64

###############################################################################
# We'll be feeding the loss with trajectories of length 25:
random_crop_len = 25

###############################################################################
# In the original paper, the authors perform one update with a batch of 64
# elements for each frame collected. Here, we reproduce the same ratio
# but while realizing several updates at each batch collection. We
# adapt our batch-size to achieve the same number of update-per-frame ratio:

batch_size = ceil_div(64 * frames_per_batch, update_to_data * random_crop_len)

replay_buffer = make_replay_buffer(
    buffer_size=buffer_size,
    batch_size=batch_size,
    random_crop_len=random_crop_len,
    prefetch=3,
    prb=prb,
)

###############################################################################
# Loss module construction
# ------------------------
#
# We build our loss module with the actor and ``qnet`` we've just created.
# Because we have target parameters to update, we _must_ create a target network
# updater.
#

gamma = 0.99
lmbda = 0.9
tau = 0.001  # Decay factor for the target network

loss_module = DDPGLoss(actor, qnet)

###############################################################################
# let's use the TD(lambda) estimator!
loss_module.make_value_estimator(ValueEstimators.TDLambda, gamma=gamma, lmbda=lmbda)

from torchrl.objectives.utils import SoftUpdate

target_net_updater = SoftUpdate(loss_module, eps=1 - tau)

###############################################################################
# Optimizer
# ~~~~~~~~~
#
# Finally, we will use the Adam optimizer for the policy and value network:

from torch import optim

optimizer_actor = optim.Adam(
    loss_module.actor_network_params.values(True, True), lr=1e-4, weight_decay=0.0
)
optimizer_value = optim.Adam(
    loss_module.value_network_params.values(True, True), lr=1e-3, weight_decay=1e-2
)
total_collection_steps = total_frames // frames_per_batch

###############################################################################
# Time to train the policy
# ------------------------
#
# The training loop is pretty straightforward now that we have built all the
# modules we need.
#

rewards = []
rewards_eval = []

# Main loop

collected_frames = 0
pbar = tqdm.tqdm(total=total_frames)
r0 = None
for i, tensordict in enumerate(collector):

    # update weights of the inference policy
    collector.update_policy_weights_()

    if r0 is None:
        r0 = tensordict["next", "reward"].mean().item()
    pbar.update(tensordict.numel())

    # extend the replay buffer with the new data
    current_frames = tensordict.numel()
    collected_frames += current_frames
    replay_buffer.extend(tensordict.cpu())

    # optimization steps
    if collected_frames >= init_random_frames:
        for _ in range(update_to_data):
            # sample from replay buffer
            sampled_tensordict = replay_buffer.sample().to(device)

            # Compute loss
            loss_dict = loss_module(sampled_tensordict)

            # optimize
            loss_dict["loss_actor"].backward()
            gn1 = torch.nn.utils.clip_grad_norm_(
                loss_module.actor_network_params.values(True, True), 10.0
            )
            optimizer_actor.step()
            optimizer_actor.zero_grad()

            loss_dict["loss_value"].backward()
            gn2 = torch.nn.utils.clip_grad_norm_(
                loss_module.value_network_params.values(True, True), 10.0
            )
            optimizer_value.step()
            optimizer_value.zero_grad()

            gn = (gn1**2 + gn2**2) ** 0.5

            # update priority
            if prb:
                replay_buffer.update_tensordict_priority(sampled_tensordict)
            # update target network
            target_net_updater.step()

    rewards.append(
        (
            i,
            tensordict["next", "reward"].mean().item(),
        )
    )
    td_record = recorder(None)
    if td_record is not None:
        rewards_eval.append((i, td_record["r_evaluation"].item()))
    if len(rewards_eval) and collected_frames >= init_random_frames:
        target_value = loss_dict["target_value"].item()
        loss_value = loss_dict["loss_value"].item()
        loss_actor = loss_dict["loss_actor"].item()
        rn = sampled_tensordict["next", "reward"].mean().item()
        rs = sampled_tensordict["next", "reward"].std().item()
        pbar.set_description(
            f"reward: {rewards[-1][1]: 4.2f} (r0 = {r0: 4.2f}), "
            f"reward eval: reward: {rewards_eval[-1][1]: 4.2f}, "
            f"reward normalized={rn :4.2f}/{rs :4.2f}, "
            f"grad norm={gn: 4.2f}, "
            f"loss_value={loss_value: 4.2f}, "
            f"loss_actor={loss_actor: 4.2f}, "
            f"target value: {target_value: 4.2f}"
        )

    # update the exploration strategy
    actor_model_explore[1].step(current_frames)

collector.shutdown()
del collector

###############################################################################
# Experiment results
# ------------------
#
# We make a simple plot of the average rewards during training. We can observe
# that our policy learned quite well to solve the task.
#
# .. note::
#   As already mentioned above, to get a more reasonable performance,
#   use a greater value for ``total_frames`` for example, 1M.

from matplotlib import pyplot as plt

plt.figure()
plt.plot(*zip(*rewards), label="training")
plt.plot(*zip(*rewards_eval), label="eval")
plt.legend()
plt.xlabel("iter")
plt.ylabel("reward")
plt.tight_layout()

