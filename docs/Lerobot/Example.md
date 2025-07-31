---
layout: default
title: Example
nav_order: 2
---

# Example

다음은 실제 예제를 통해서 Lerobot의 Immitation learning을 하는 방법에 대해서 알아보겠습니다.
그 전에 먼저 Leader 로봇팔을 통해서 Follower 로봇팔을 조작하는 teleoperate 예시부터 진행해 보겠습니다.


## Teleoperate
Teleoperate의 경우 leader암을 통해서 follower암을 조작할 수 있도록 합니다. Teleoperate 명령을 통해 Leader 로봇팔의 움직임을 Follower 로봇팔이 실시간으로 복제하도록 설정할 수 있습니다. 명령어는 다음과 같습니다.

```bash
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=cvr_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=cvr_leader_arm
```

사용하는 argument는 다음과 같습니다:

<pre>
--robot.type    # Follower 로봇의 모델 타입을 지정합니다. 예) so101_follower
--robot.port    # Follower 로봇이 연결된 직렬 포트 경로를 지정합니다. 예) /dev/ttyACM0
--robot.id      # 보정(calibration) 파일을 저장할 식별자(ID)입니다.
--teleop.type   # Teleop 장치(Leader 암)의 모델을 지정합니다. 예) so101_leader
--teleop.port   # Teleop 장치가 연결된 포트 경로를 지정합니다. 예) /dev/ttyACM1
--teleop.id     # Teleop 장치의 식별자(ID)로, 모든 단계에서 동일하게 사용해야 합니다.
</pre>

또한 텔레오퍼레이션에 카메라 피드를 추가하면 로봇의 조작과 동시에 실시간 영상도 함께 확인할 수 있습니다. 특히 물체의 위치나 자세를 보면서 조작할 때 매우 유용합니다. 다음은 teleoperate 예제에 카메라를 추가하는 예시입니다.
주요 구조는 위의 teleoperate명령어와 거의 동일하지만 사용하는 카메라를 명령어에 추가시켜야 합니다. 명령어는 다음과 같습니다:

```bash
python -m lerobot.teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=cvr_follower_arm \
    --robot.cameras="{ front: {type: intelrealsense, width: 640, height: 480, fps: 15, serial_number_or_name: 223322300015}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 15}}"
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=cvr_leader_arm
```

camera를 사용하는 경우 intelrealsense, opencv, phone cam 등 이렇게 3가지 예시가 있을 수 있는데 핸드폰 카메라를 사용하는 경우 v4l2loopback-dkms 설치 오류로 인해 진행하지 못했습니다. 만약 이외의 카메라를 사용한다면 위의 예시를 참고하여 명령어를 작성해주시면 될 것 같습니다. 또한 카메라 관련 링크를 하단에 첨부해 두겠습니다.

[Camera setup](https://huggingface.co/docs/lerobot/cameras?use+phone=Linux#setup-cameras)

## Record & Replay

Record 단계는 사람이 텔레오퍼레이션을 통해 수행한 작업을 기반으로 데이터셋을 생성하는 단계입니다. 아래는 lerobot을 이용해 특정 작업(예: Grab the orange cube and put into black box)을 수행하며 데이터를 기록하는 예시입니다.

```bash
python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=cvr_follower_arm \
  --robot.cameras="{ front: {type: intelrealsense, width: 640, height: 480, fps: 15, serial_number_or_name: 223322300039}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 15}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=cvr_leader_arm \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/record-test-box \
  --dataset.num_episodes=50 \
  --dataset.single_task="Grab the orange cube and put into black box"
```

record의 경우 기존의 teleoperate명령어와 다르게 dataset, display_data와 같은 인자가 추가됩니다. 각 인자는 다음과 같습니다.

<pre>
--display_data          # 각 모터, 카메라의 정보를 컴퓨터에서 확인할 수 있습니다.
--dataset.repo_id       # HuggingFace에 업로드될 데이터셋 저장 위치를 지정합니다.
--dataset.num_episodes  # 몇 개의 에피소드를 기록할지 지정합니다. (약 50개)
--dataset.single_task   # imitation learning에 사용될 자연어 task 명령어를 지정.
</pre>

이렇게 record한 데이터셋은 replay를 통해서 저장했던 action을 그대로 follower암에서 재현할 수 있습니다. 명령어는 다음과 같습니다.

```bash
python -m lerobot.replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=cvr_follower_arm \
    --dataset.repo_id=${HF_USER}/record-test-box \
    --dataset.episode=13
```

여기에서 episode를 지정하여 action이 잘 저장되고 실행되는지를 확인할 수 있으며 만약 상자를 잡는 task라면 상자 위치를 record할 때 사용했던 위치와 완전 동일하게 배치해야지 정상적으로 상자를 집어 원하는 task를 수행 할 수 있습니다.

## train

다음은 train입니다. train의 경우 2가지 예시로 나눌 수 있습니다. 먼저 아무런 학습이 되지 않은 네트워크를 가져와서 immitation learning을 시키는 경우와 smolVLA와 같이 사전 학습된 LLM모델을 불러와 finetuning시키는 두가지 경우가 있습니다.


### immitation learning
먼저 immitation learning입니다. record했던 데이터셋을 기반으로 immitation learning을 진행하며 초기화된 정책(--policy.type=act)을 사용하여 데이터셋으로부터 행동을 모방하도록 학습합니다.

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=${HF_USER}/record-test-box0 \
    --policy.type=act \   
    --output_dir=outputs/train/act_test \   
    --job_name=act_test \   
    --policy.device=cuda \   
    --wandb.enable=false \ 
    --policy.repo_id=junho232
```

여기에서 immitation learning을 위한 추가 argument는 다음과 같습니다.,

<pre>
--policy.type    # 사전 학습 없이, act 기반 기본 모델을 사용합니다.
--output_dir     # 학습 중 생성되는 로그, 체크포인트 등을 저장하는 경로입니다.
--job_name       # 학습 job의 이름입니다. wandb에 사용됩니다.
--wandb.enable   # wandb 사용여부를 결정합니다.
--policy.repo_id # 학습 완료 후 HuggingFace에 policy를 업로드할 위치입니다
</pre>

여기에서 사용되는 policy.type은 act알고리즘을 사용하게 되는데, 이외에도 diffusion, pi0, pi0fast, smolvla등 다양한 알고리즘들이 존재합니다. 테스트 결과 act알고리즘이 데이터셋 50~60개에서 가장 잘 작동하는 것을 확인하였고 개인적으로 act알고리즘의 논문을 한번 확인해보고 어떻게 작동하는지 알아두면 좋을 것 같습니다. 

[ACT 논문](https://arxiv.org/abs/2304.13705)

### SmolVLA finetuning
이 방식은 사전 학습된 LLM 기반 정책인 smolvla_base를 불러와서 imitation dataset으로 fine-tuning하는 예시입니다.

```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/record-test-box \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_record_test_policy \
  --job_name=record_test_training \
  --policy.device=cuda \
  --policy.repo_id=${HF_USER}/my_record_test_policy \
  --wandb.enable=true
```

