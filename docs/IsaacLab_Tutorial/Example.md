---
layout: default
title: Example
nav_order: 3
---

# Example

다음은 실제 예제를 통해서 Direct와 Manager-based의 차이를 알아보도록 하겠습니다.Direct 방식은 하나의 클래스 내에서 환경의 모든 구성 요소(보상 함수, 관측, 초기화, 종료 조건, 행동 적용 등)을 직접 구현하는 방식입니다. 반면 Manager 기반 방식은 이러한 구성 요소를 각각의 모듈로 분리하여 구성하며, 재사용성과 구성 변경의 유연성에 초점을 맞춥니다.

Direct 방식에서는 관측, 보상, 초기화 같은 환경의 동작을 사용자가 직접 함수로 만들어야 합니다. 예를 들어, 관측을 계산하는 함수(_get_observations)나 보상을 계산하는 함수(_get_rewards)를 직접 작성해서 환경 클래스에 넣어야 합니다. 

반면 Manager-based 방식은 ObservationsCfg, RewardsCfg, EventCfg, TerminationsCfg 등의 구성 클래스를 조합하여 환경을 정의합니다. 각 Manager는 특정 역할만을 수행하며, 서로 독립적으로 개발하고 수정할 수 있습니다. 이는 실험 구성을 빠르게 변경하거나 여러 task 간 공통 로직을 재사용할 수 있다는 장점이 있습니다.

이제 두 방식 각각의 코드 예제와 그 구성 요소를 살펴보며, 어떤 방식이 사용자의 프로젝트에 더 적합할지 판단해보시기 바랍니다.


## Direct task

예제에서 제공된 CartpoleEnv 클래스는 Direct 방식으로 작성된 환경입니다.

주요 구조는 다음과 같습니다:

<pre>
 - __init__(): 로봇의 관절 인덱스와 상태 버퍼를 설정합니다.
 - _setup_scene(): 로봇과 바닥, 조명을 씬에 배치하고 물리 복제를 설정합니다.
 - _pre_physics_step()과 _apply_action(): 행동 벡터를 받아 실제 물리 시뮬레이션에 적용합니다.
 - _get_observations(): 현재 관절 상태로부터 관측 벡터를 구성합니다.
 - _get_rewards(): 지정된 조건에 따라 보상을 계산합니다.
 - _get_dones(): 카트나 막대의 상태가 유효 범위를 넘었는지 확인하여 종료 조건을 반환합니다.
 - _reset_idx(): 초기 상태를 무작위로 샘플링하여 재설정합니다.
</pre>

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

Direct cartpole의 실행 코드는 다음과 같습니다.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-Direct-v0
```

이 코드를 실행하는 영상을 하단에 첨부합니다.

<video width="640" height="360" controls>
  <source src="assets/video/스크린캐스트 05-14-2025 03:09:18 PM.webm" type="video/webm">
  Your browser does not support the video tag.
</video>


## Manager-based task

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

Manager-based cartpole의 실행 코드는 다음과 같습니다.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Cartpole-v0
```

이 코드를 실행하는 영상을 하단에 첨부합니다.

<video width="640" height="360" controls>
  <source src="assets/video/스크린캐스트 05-14-2025 03:23:58 PM.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

또한 Manager-based를 실행할 경우 각 클래스 매니저에 대한 정보값이 코드를 실행할 때 보여집니다. 이는 하단에 첨부한 동영상과 같이 실행 터미널 위쪽에서 확인할 수 있습니다.

<video width="640" height="360" controls>
  <source src="assets/video/스크린캐스트 05-14-2025 03:30:46 PM.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

이는 Direct task에 비해 현재 학습 환경에 대한 정보도 얻기 좋으며 좀 더 체계적으로 학습을 진행 할 수 있습니다. 다만 이렇게 Manager-based 환경을 꾸미기 위해서 상속받은 ManagerBasedRLEnvCfg을 정확하게 이해하고 사용해야 된다는 점에서 꽤나 오랜 시간을 소요하게 되고 복잡성이 증가합니다. 


그렇기에 새로운 환경을 처음부터 구축해 나가는 것은 Direct 환경을 추천드리고 기존에 있던 환경에 terrain을 변경하거나 command나 reward를 변경하는 등 모듈 수준에서 환경을 꾸미게 된다면 Manager-based 환경을 추천드립니다.

 

