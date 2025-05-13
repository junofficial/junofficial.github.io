---
layout: default
title: Isaac Lab Tutorial
nav_order: 1
---

# Isaac Lab Tutorial

<video width="640" height="360" controls>
  <source src="assets/img/스크린캐스트 05-02-2025 05:30:24 PM.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

Isaac Lab 튜토리얼에 오신 것을 환영합니다!

이 문서는 Isaac Lab을 보다 빠르고 쉽게 시작할 수 있도록 제작되었습니다. Isaac Lab 사용이 처음이신 분들을 위해 설치 및 실행에 필요한 단계들을 정리해 놓았으며 quadcopter 예제를 직접 실행하는 것 까지 문서를 정리해 두었습니다.


## Install
먼저 Isaac Lab을 사용하려면 Omniverse Isaac Sim을 설치해야 합니다.

Isaac Lab은 NVIDIA의 시뮬레이션 엔진인 Isaac Sim 위에서 실행되며,
로봇 물리 시뮬레이션, 센서 렌더링, 강화 학습 환경 실행과 같은 핵심 기능을 Isaac Sim에 의존합니다.

### Isaac Sim install

먼저, Isaac Lab 프로젝트를 위한 Python 환경을 설정해야 합니다.
Python 3.10을 기반으로 한 conda 가상 환경을 사용하여 개발 환경을 구성하겠습니다.
```bash
conda create -b isaaclab python=3.10
conda activate isaaclab
```

다음으로, 사용 중인 시스템의 CUDA 버전에 맞춰 CUDA를 지원하는 PyTorch 2.5.1을 설치합니다.

현재 CUDA 11을 사용중이시라면
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

현재 CUDA 12를 사용중이시라면
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

pip을 사용해 Isaac Sim 패키지를 설치하기 위해, 먼저 pip을 최신 버전으로 업그레이드해야 합니다.
```bash
pip install --upgrade pip
```

그다음, Isaac Sim을 설치합니다.
```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

다음 코드를 사용하여 Isaac Sim이 정상적으로 실행되는지 확인하세요:
```bash
isaacsim
```

Isaac Sim을 처음 실행할 때, 라이선스 동의(EULA) 창이 나타납니다.
계속 진행하려면 "yes"를 입력하여 동의해야 합니다.


```bash
Do you accept the EULA? (Yes/No): Yes
```

### Isaac Lab install

먼저 git을 통해서 isaac lab을 설치해 줍니다.

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
```

이후 cmake의존성을 설치해 줍니다.

```bash
sudo apt install cmake build-essential
```

설치 완료된 이후 IsaacLab폴더에 들어가서 프로젝트에서 제공하는 shell을 통해서 IsaacLab에 필요한 pip들을 설치해줍니다.

```bash
./isaaclab.sh --install
```

이후 Isaac Lab이 올바르게 설치됬는지 쉘 스크립트를 통해서 create_empty.py를 실행해 줍니다.

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

이 shell 코드는 밑의 파이썬 실행 코드와 같은 역할을 합니다.

```bash
python scripts/tutorials/00_sim/create_empty.py
```

## Things to know

이 목차에서는 Isaac Lab을 사용하기 전에 알고 있으면 좋을 내용들에 대해서 정리해 놓았습니다. 실행이 우선이면 example을 통해 예제를 확인하시고 

### Available environments

Isaac Lab은 다양한 로봇 제어 및 강화 학습 실험을 위한 환경들을 제공합니다. 각 환경은 특정 로봇 유형, 제어 목적, 시뮬레이션 설정에 따라 구성되어 있으며 여러 카테고리로 나뉩니다.

이와 관련한 문서의 링크를 하이퍼링크를 통해 남깁니다.

[Isaac Lab에서 사용 가능한  Environment](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html).

또한 IsaacLab폴더에서 shell 스크립트를 통해서 사용가능한 env목록들을 확인할 수 있습니다.

```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```

이러한 환경을 강화학습을 통해 학습시키는 명령어는 다음과 같습니다.

```bash
python scripts/reinforcement_learning/<specific-rl-library>/train.py --task=<Task-Name>
```

specific-rl-library 부분에 사용 가능한 강화학습 라이브러리를 선택해주고 (예 : rsl_rl) 원하는 환경을 Task-Name 부분에 작성하면 됩니다. (예 : Isaac-Cartpole-Direct-v0)

또한 zero-action 혹은 random-action을 통해 환경을 확인하고 싶다면 하단의 명령어를 사용하면 됩니다.

```bash
python scripts/zero_agent.py --task=<Task-Name>
```

```bash
python scripts/random_agent.py --task=<Task-Name>
```

### Task design

Isaac Lab에서의 Task는 특정 에이전트(로봇)에 대한 관측(observations)과 행동(actions)을 정의하는 환경(environment)으로 구성됩니다. 이러한 환경은 에이전트에게 현재 상태를 제공하고, 에이전트의 행동을 시뮬레이션을 통해 실행합니다. Isaac Lab은 이러한 Task를 설계하기 위해 Manager-based와 direct 같은 두 가지 주요 워크플로우를 제공합니다:

#### Manager-based 워크플로우

![steps screenshot](assets/img/manager-based-light.svg)

Manager 기반 환경은 Task를 여러 개의 독립적인 구성 요소(Managers)로 분해하여 모듈화된 구현을 지원합니다. 각 Manager는 보상 계산, 관측 처리, 행동 적용, 무작위화 등 특정 기능을 담당하며, 사용자는 각 Manager에 대한 구성 클래스를 정의합니다. 이러한 Managers는 envs.ManagerBasedEnv를 상속하는 환경 클래스에 의해 조정되며, 다양한 구성을 쉽게 교체하거나 확장할 수 있습니다.

**장점:**
  - 모듈화된 설계로 구성 요소의 재사용 및 유지보수에 유리
  - 다양한 구성을 실험하고 검증하는데 유리
  - 협업 시 각 구성 요소를 독립적으로 개발하고 통합할 수 있음

**단점:**

 - 구현의 복잡성이 증가할 수 있으며, 각 Manager 간의 상호작용을 명확히 이해해야 사용 가능
 - 세밀한 제어가 필요한 경우에는 Direct에 비해 제한적

Manager-based로 작성된 reward함수는 다음과 같습니다.

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

#### Direct 워크플로우

![steps screenshot](assets/img/direct-based-light.svg)

Direct 워크플로우는 전통적인 환경 구현 방식과 유사하게, 단일 클래스에서 보상 함수, 관측 처리, 리셋 조건 등을 직접 구현합니다. 이 접근 방식은 envs.DirectRLEnv 또는 envs.DirectMARLEnv를 상속하여 사용하며, Manager 클래스를 사용하지 않고 전체 환경 로직을 직접 제어할 수 있습니다.

**장점:**
  - 환경 로직에 대한 완전한 제어가 가능하여 복잡한 로직 구현에 적합
  - PyTorch JIT 또는 Warp와 같은 최적화 기법을 활용하여 성능을 향상시킬 수 있음
  - IsaacGymEnvs 또는 OmniIsaacGymEnvs에서 마이그레이션하는 사용자에게 친숙한 구조

**단점:**

 - 구현의 재사용성과 모듈화가 제한적
 - 구성 요소의 교체나 확장이 Manager 기반 워크플로우에 비해 어려움
 

Direct로 작성된 reward함수는 다음과 같습니다. 
 
```python
@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
```

## Example

다음은 실제 예제를 통해서 Direct와 Manager-based의 차이를 알아보도록 하겠습니다.Direct 방식은 하나의 클래스 내에서 환경의 모든 구성 요소(보상 함수, 관측, 초기화, 종료 조건, 행동 적용 등)을 직접 구현하는 방식입니다. 반면 Manager 기반 방식은 이러한 구성 요소를 각각의 모듈로 분리하여 구성하며, 재사용성과 구성 변경의 유연성에 초점을 맞춥니다.

Direct 방식에서는 관측, 보상, 초기화 같은 환경의 동작을 사용자가 직접 함수로 만들어야 합니다. 예를 들어, 관측을 계산하는 함수(_get_observations)나 보상을 계산하는 함수(_get_rewards)를 직접 작성해서 환경 클래스에 넣어야 합니다. 

반면 Manager-based 방식은 ObservationsCfg, RewardsCfg, EventCfg, TerminationsCfg 등의 구성 클래스를 조합하여 환경을 정의합니다. 각 Manager는 특정 역할만을 수행하며, 서로 독립적으로 개발하고 수정할 수 있습니다. 이는 실험 구성을 빠르게 변경하거나 여러 task 간 공통 로직을 재사용할 수 있다는 장점이 있습니다.

이제 두 방식 각각의 코드 예제와 그 구성 요소를 살펴보며, 어떤 방식이 사용자의 프로젝트에 더 적합할지 판단해보시기 바랍니다.


### Direct task

예제에서 제공된 CartpoleEnv 클래스는 Direct 방식으로 작성된 환경입니다.

주요 구조는 다음과 같습니다:

 - __init__(): 로봇의 관절 인덱스와 상태 버퍼를 설정합니다.
 - _setup_scene(): 로봇과 바닥, 조명을 씬에 배치하고 물리 복제를 설정합니다.
 - _pre_physics_step()과 _apply_action(): 행동 벡터를 받아 실제 물리 시뮬레이션에 적용합니다.
 - _get_observations(): 현재 관절 상태로부터 관측 벡터를 구성합니다.
 - _get_rewards(): 지정된 조건에 따라 보상을 계산합니다.
 - _get_dones(): 카트나 막대의 상태가 유효 범위를 넘었는지 확인하여 종료 조건을 반환합니다.
 - _reset_idx(): 초기 상태를 무작위로 샘플링하여 재설정합니다.

**class CartpoleEnvCfg**

이 클래스는 configuration 클래스로 환경의 동작 주기, 시뮬레이션 설정, 로봇 정보, 보상 관련 스케일 등을 정의합니다.

```python
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # 환경에 대한 설정 클래스
    decimation = 2                          # 시뮬레이션 스텝 중 몇 번에 한 번 행동 적용할지 결정
    episode_length_s = 5.0                  # 한 에피소드의 최대 길이 (초 단위)
    action_scale = 100.0                    # 행동 값에 곱해지는 스케일
    action_space = 1                        # 행동 공간의 크기
    observation_space = 4                   # 관측 벡터의 길이
    state_space = 0                         # 사용하지 않음

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"        # 카트 관절 이름
    pole_dof_name = "cart_to_pole"          # 폴 관절 이름

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    max_cart_pos = 3.0                      # 카트가 벗어나면 에피소드 종료
    initial_pole_angle_range = [-0.25, 0.25]  # 폴 초기 각도 범위

    # 보상 함수 항목별 스케일
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
```
**__init__(...)**

이 함수는 환경 인스턴스를 초기화하는 생성자입니다.
주어진 설정(config)을 바탕으로 환경 기본 요소들을 구성하며, 로봇 아티큘레이션에서 카트와 폴의 관절 이름에 해당하는 인덱스를 찾아 저장합니다.

```python
class CartpoleEnv(DirectRLEnv):
    # 환경 클래스 정의
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스를 찾고 저장
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)

        self.action_scale = self.cfg.action_scale
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel
```

**_setup_scene(...)**

이 함수는 시뮬레이션 내에 로봇과 환경 요소를 배치하는 역할을 합니다.
로봇 아티큘레이션을 생성하고, 지면 평면을 추가하며, 여러 환경 인스턴스를 클론하여 배치합니다.
이 과정에서 각 환경은 물리적으로 분리되어 있으며, 복수의 학습 환경을 병렬로 시뮬레이션할 수 있도록 구성됩니다.

**_pre_physics_step(...)**

이 함수는 물리 시뮬레이션이 실행되기 전 호출되어 policy로 부터 받은 action을 물리적으로 적용시키기 위해서 스케일링 등과 같이 조정되는 단계입니다.

**_apply_action(...)**

이 함수는 _pre_physics_step(...)에서 조정된 action값을 실제 시뮬레이션에 적용하는 단계입니다. 

```python
    def _setup_scene(self):
        # 로봇 추가 및 환경 설정
        self.cartpole = Articulation(self.cfg.robot_cfg) # 관절 요소 추가
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg()) # 지면 추가
        self.scene.clone_environments(copy_from_source=False) # Scene에 환경 복제(병렬화)
        self.scene.articulations["cartpole"] = self.cartpole # Scene에 등록

        # 조명 추가
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 물리 엔진에 전달하기 전에 행동 스케일링
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        # 행동을 카트에 힘으로 적용
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
```

**_get_observations(...)**

이 함수는 에이전트의 현재 관측값을 받아오는 함수입니다. Obs의 경우 cartpole에서는 pole의 속도와 각도, 카드의 위치와 속도를 obs로 결합하여 사용합니다.
        
```python
    def _get_observations(self) -> dict:
        # 관측 벡터를 구성하여 반환
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
```

**_get_rewards(...)**

이 함수는 현재 상태에서의 보상을 계산하는 함수입니다. 보상항목은 생존 보상, 종료 패널티, 막대의 수직도, 카드 속도, 막대 각속도 등으로 구성되어 있으며 self.cfg는 초기에 CarpoleEnvCfg에서 설정한 reward scales와 보상을 계산하기 위한 현재 agent의 상태들을 compute_rewards에서 계산하게 됩니다.


```python
    def _get_rewards(self) -> torch.Tensor:
        # 보상 함수 계산
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward
```

**compute_rewards(...)**

compute rewards에서 보상은 다음과 같이 계산됩니다.

 - R = R_alive + R_termination + R_pole_pos + R_cart_vel + R_pole_vel
 - R_alive = w_alive × (1 - d)
 - R_termination = w_termination × d
 - R_pole_pos = w_pole_pos × sum(p_i^2)
 - R_cart_vel = w_cart_vel × sum(|v_cart_i|)
 - R_pole_vel = w_pole_vel × sum(|v_pole_i|)

여기서,
 - d는 에피소드 종료 여부 (종료: 1, 생존: 0)
 - p_i는 막대의 관절 위치(각도)
 - v_cart_i, v_pole_i는 각각 카트와 폴의 관절 속도
 - w_*는 각 보상 항목에 대한 스케일 값입니다 (rew_scale_*)

```python
@torch.jit.script
def compute_rewards(
    # 각 보상 항목 계산 및 총합
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
```

**_get_dones(...)**

이 함수는 에피소드가 종료되었는지의 여부를 판단합니다. 종료 조건은 두가지로 다음과 같습니다:

 - done = done_pos or done_time
 - done_time = (t >= T_max)
 - done_pos = (|x_cart| > x_max) or (|theta_pole| > π / 2)

```python
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 종료 조건 평가 (시간 초과 또는 위치 초과)
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out
```

**_reset_idx(...)**

이 함수는 환경들을 초기화 하는 함수입니다. 관절의 초기 위치와 속도를 설정하며 폴의 경우에는 범위내에서 무작위로 샘플링됩니다.

```python
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # 선택된 환경 인덱스를 초기화
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```


### Manager-based task

이번 예제는 Cartpole task를 manager-based 방식으로 작성한 환경입니다. Manager-based task는 Isaac Lab에서 제공하는 환경 모듈화 방식을 따릅니다. 각 기능(보상, 관측, 초기화, 종료 조건 등)을 독립적인 Manager 클래스로 나누고, 이들을 하나의 설정(Config class)으로 통합합니다. 구성 요소 간 의존성이 줄어들기 때문에 복잡한 시나리오나 다수의 환경을 동시에 개발하고자 할 때 유리한 구조입니다.

주요 구조는 다음과 같습니다:

 - CartpoleSceneCfg: 환경 내 지형, 조명, 로봇을 정의합니다.
 - ActionsCfg: 어떤 방식으로 행동을 적용할지 설정합니다.
 - ObservationsCfg: 어떤 관측값을 policy에 전달할지 정의합니다.
 - EventCfg: 리셋 시 어떤 초기화를 적용할지 설정합니다.
 - RewardsCfg: 보상을 구성하는 항목과 스케일을 정의합니다.
 - TerminationsCfg: 종료 조건들을 정의합니다.

이러한 구성들은 ManagerBasedRLEnvCfg를 상속한 환경 설정 클래스(CartpoleEnvCfg)에서 모두 통합되며, 실행 시 ManagerBasedEnv가 이 구성들을 읽어 전체 시뮬레이션 루프를 관리합니다.

**CartpoleSceneCfg**

이 클래스는 시뮬레이션 내의 물리적 배치를 정의합니다. 지면(ground), 로봇(robot), 조명(light)을 설정합니다.


```python
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # 지면 설정: 넓은 평면을 생성
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # 카트폴 로봇 설정: 환경별로 prim_path 지정
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 돔 라이트 설정: 밝기와 색 지정
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

**ActionsCfg**

이 클래스는 joint_effort를 통해 카트에 직접적인 힘을 가합니다.

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 카트 관절에 joint effort 방식의 제어 적용 (힘 기반 제어)
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
```

**ObservationsCfg**

이 클래스에서는 정책에 입력되는 obs 구성을 정의하는 PolicyCfg로 이루어져 있으며 관절 위치 및 속도를 상대적인 값으로 policy에 전달하게 됩니다.

```python
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 상대 관절 위치와 속도를 관측값으로 사용
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            # 관측값 손상 비활성화 및 항목 결합 설정
            self.enable_corruption = False
            self.concatenate_terms = True

    # policy라는 이름의 관측 그룹으로 지정
    policy: PolicyCfg = PolicyCfg()
```

**EventCfg**

이 클래스는 이벤트를 정의하는 클래스로 현 task에서는 카트 및 폴의 위치와 속도를 일정 범위에서 무작위로 초기화합니다.

```python
@configclass
class EventCfg:
    """Configuration for events."""

    # 카트 관절 초기화: 위치와 속도 범위 내 무작위 샘플링
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # 폴 관절 초기화: 각도 및 각속도 범위 내 무작위 샘플링
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )
```

**RewardsCfg**

보상 함수는 다음과 같은 총합으로 구성됩니다:

R = R_alive + R_termination + R_pole_pos + R_cart_vel + R_pole_vel

각 항목은 다음과 같이 정의됩니다:

- 생존 보상 (계속 살아있을 경우 1점 부여):  
  R_alive = w_alive × (1 - d)  
  - d: 종료 여부 (종료 시 d = 1, 생존 시 d = 0)  
  - w_alive = 1.0

- 종료 페널티 (에피소드가 끝나면 벌점):  
  R_termination = w_termination × d  
  - w_termination = -2.0

- 폴 각도 오차 보상 (막대를 수직으로 유지):  
  R_pole_pos = w_pole_pos × ||θ_pole - θ_target||²  
  - θ_target = 0 (직립 상태)  
  - w_pole_pos = -1.0

- 카트 속도 줄이기 보상:  
  R_cart_vel = w_cart_vel × ||v_cart||₁  
  - w_cart_vel = -0.01

- 폴 각속도 줄이기 보상:  
  R_pole_vel = w_pole_vel × ||ω_pole||₁  
  - w_pole_vel = -0.005


```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 생존 보상: 에피소드 지속에 대해 보상
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # 종료 패널티: 실패 시 페널티 부여
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # 폴 각도를 수직으로 유지할수록 보상 증가
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # 카트 속도 최소화 보상
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])}
    )
    # 폴의 각속도 최소화 보상
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])}
    )
```

** TerminationsCfg **

이 클래스는 종료 조건을 정의하는 클래스로 현 task에서는 시간 초과 또는 카트가 주어진 위치 범위를 벗어난 경우 종료합니다.

```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 일정 시간 초과 시 종료
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 카트가 제한된 위치를 벗어날 경우 종료
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )
```

**CartpoleEnvCfg**

이 클래스는 위에 정의한 scene, observation, action, reward, event, termination 구성 요소를 통합하여 전체 환경 설정을 완성합니다.

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene 설정: 지형, 조명, 로봇 포함
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    # 관측 설정
    observations: ObservationsCfg = ObservationsCfg()
    # 행동 설정
    actions: ActionsCfg = ActionsCfg()
    # 초기화 이벤트 설정
    events: EventCfg = EventCfg()
    # 보상 설정
    rewards: RewardsCfg = RewardsCfg()
    # 종료 조건 설정
    terminations: TerminationsCfg = TerminationsCfg()

    # 시뮬레이션, 렌더링, 뷰어 설정
    def __post_init__(self) -> None:
        self.decimation = 2                               # 시뮬레이션 스텝마다 행동 적용 빈도
        self.episode_length_s = 5                         # 에피소드 길이 (초)
        self.viewer.eye = (8.0, 0.0, 5.0)                 # 카메라 위치 설정
        self.sim.dt = 1 / 120                             # 시뮬레이션 타임스텝
        self.sim.render_interval = self.decimation       # 렌더링 빈도 설정
```
## TIPS

### Urdf to usd

### Creating new env





