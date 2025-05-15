---
layout: default
title: TIPS
nav_order: 4
---

# TIPS

마지막으로 Isaac Lab을 사용해보면서 알고 있으면 좋은 내용들을 정리했습니다. 필수적으로 읽어야 하는 부분은 아니며, Isaac Lab을 사용하다 막히는 부분이 생겼을 때 아래 TIPS 항목들을 참고하시면 도움이 될 수 있습니다.

혹시 Isaac Lab을 사용하는 도중 문제가 발생하거나 문서에 추가되었으면 하는 내용이 있다면, 아래 이메일로 연락주시면 검토 후 TIPS에 반영하도록 하겠습니다.

📩 문의: junho1458@gmail.com


## Available argument

<pre>
🔧 주요 Argument 설명

Argument                         설명
------------------------------  -----------------------------------------------------------
-h, --help                      도움말 메시지 출력
--video                         학습 중 비디오 저장 활성화
--video_length VIDEO_LENGTH     저장할 비디오 길이 (프레임 단위)
--video_interval VIDEO_INTERVAL 몇 iteration마다 비디오를 저장할지
--num_envs NUM_ENVS             병렬 환경 개수 설정 (예: 4096)
--task TASK                     학습할 환경 이름 (예: Isaac-Cartpole-v0)
--seed SEED                     랜덤 시드 값 (재현 가능성 확보용)
--max_iterations MAX_ITERATIONS 최대 학습 iteration 수
--experiment_name EXPERIMENT_NAME 실험 디렉토리 이름 설정 (logs/ 하위 경로)
--run_name RUN_NAME             동일 실험 내 여러 run을 구분하기 위한 이름
--resume RESUME                 학습 중단 후 이어서 학습할지 여부 (True or False)
--load_run LOAD_RUN             기존 run에서 모델을 로드할 경우 해당 run 이름
--checkpoint CHECKPOINT         로드할 checkpoint 번호
--logger {wandb,tensorboard,neptune} 사용할 로깅 도구 지정
--log_project_name LOG_PROJECT_NAME 로깅 툴에서 사용할 프로젝트 이름
--headless                      GUI 없이 headless 모드 실행 (서버에서 실행 시 사용)
--livestream {0,1,2}            livestream 시각화 모드 설정 (0: 없음, 1: minimal, 2: full)
--enable_cameras                환경에서 카메라 사용 여부
--device DEVICE                 연산 디바이스 설정 (cuda, cpu 등)
--verbose                       출력 상세 정보 활성화
--info                          환경의 추가 정보 출력 여부
--experience EXPERIENCE         사전 정의된 경험 설정 사용 (예: curriculum)
--kit_args KIT_ARGS             Omniverse Kit 런타임 인자 전달용 (사용 거의 없음)
</pre>



## Urdf to usd

## Creating new env

## Adding assets

## Changing RL config

## go2 isaac gym parkour

## 논문 돌린것들 몇개

## pybullet drone

## 르로봇

## Cuda 설치 방법
