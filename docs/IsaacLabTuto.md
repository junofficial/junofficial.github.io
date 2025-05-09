---
layout: default
title: Isaac Lab Tutorial
nav_order: 1
---

# Isaac Lab Tutorial

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

Isaac Lab에서의 Task는 특정 에이전트(로봇)에 대한 관측(observations)과 행동(actions)을 정의하는 환경(environment)으로 구성됩니다. 이러한 환경은 에이전트에게 현재 상태를 제공하고, 에이전트의 행동을 시뮬레이션을 통해 실행합니다. Isaac Lab은 이러한 Task를 설계하기 위한 두 가지 주요 워크플로우를 제공합니다:

#### Manager 기반 워크플로우

![steps screenshot](assets/img/direct-based-light.svg)

Manager 기반 환경은 Task를 여러 개의 독립적인 구성 요소(Managers)로 분해하여 모듈화된 구현을 촉진합니다. 각 Manager는 보상 계산, 관측 처리, 행동 적용, 무작위화 등 특정 기능을 담당하며, 사용자는 각 Manager에 대한 구성 클래스를 정의합니다. 이러한 Managers는 envs.ManagerBasedEnv를 상속하는 환경 클래스에 의해 조정되며, 다양한 구성을 쉽게 교체하거나 확장할 수 있습니다.
## Example

### Direct task

### Manager-based task

## TIPS

### Urdf to usd

### Creating new env




