---
layout: default
title: Isaac Lab Tutorial
nav_order: 2
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

#### Verifying the installation

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



