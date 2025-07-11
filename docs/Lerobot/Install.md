---
layout: default
title: What is All This? (FAQs)
nav_order: 1
---

# Install
 
## Hardware install
Lerobot 하드웨어 설치의 경우 Hugging face 공식 홈페이지에서 자세하게 설명되어 있습니다.

[Hugging face 하드웨어 설치](https://huggingface.co/docs/lerobot/so101)

하드웨어를 설치할 때 주의해야 할 부분들은 다음과 같습니다.

 1. 설치하면서 서보모터를 돌리게 된다면 나중에 calibration 할 때 모터의 magnitute가 작동할 수 있는 범위가 넘어가서 calibration이 불가능 할 수 있습니다. 설치할 때에는 서보모터를 최대한 돌리지 않고 진행하시는 것이 좋습니다.
 2. 또한 calibration을 진행하면서 firmware의 버전이 맞지 않아 오류가 발생할 수 있습니다. 해결 방법은 이후에 TIPS에 작성해 두도록 하겠습니다.
 
## Software install

먼저 lerobot github를 local에 설치해주셔야 합니다. 명령어는 다음과 같습니다.
 
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

이후 Lerobot 프로젝트를 위한 Python 환경을 설정해야 합니다.
Python 3.10을 기반으로 한 conda 가상 환경을 사용하여 개발 환경을 구성하겠습니다.

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
```

다음으로, 학습 중 카메라 사용을 위해서 ffmpeg를 추가로 설치해 주셔야 합니다.

```bash
conda install ffmpeg -c conda-forge
```

이후 lerobot 폴더에 들어가 있는 상태로 lerobot 구성 요소들을 설치해주시면 됩니다.

```bash
pip install -e .
```

## Robot setup

하드웨어와 소프트웨어를 모두 설치하였다면 로봇을 각 모터에 모터번호를 지정하고 calibrate을 진행해야 합니다.

먼저 2개의 모터 드라이버에 전원과 USB를 연결하고 터미널에서 각 모터드라이버의 포트번호를 확인합니다.

```bash
python -m lerobot.find_port
```

이 명령어를 치면 꽤나 많은 USB포트를 확인할 수 있는데 여기에서 모터드라이버의 USB를 하나 제거하고 엔터를 누르면 그 USB포트의 이름을 알 수 있습니다. 예시는 다음과 같습니다.

```bash
(lerobot) cvr@cvr1:~/Desktop/lerobot$ python -m lerobot.find_port
Finding all available ports for the MotorsBus.
Ports before disconnecting: ['/dev/tty', '/dev/tty0', '/dev/tty1', '/dev/tty2', '/dev/tty3', '/dev/tty4', '/dev/tty5', '/dev/tty6', '/dev/tty7', '/dev/tty8', '/dev/tty9', '/dev/tty10', '/dev/tty11', '/dev/tty12', '/dev/tty13', '/dev/tty14', '/dev/tty15', '/dev/tty16', '/dev/tty17', '/dev/tty18', '/dev/tty19', '/dev/tty20', '/dev/tty21', '/dev/tty22', '/dev/tty23', '/dev/tty24', '/dev/tty25', '/dev/tty26', '/dev/tty27', '/dev/tty28', '/dev/tty29', '/dev/tty30', '/dev/tty31', '/dev/tty32', '/dev/tty33', '/dev/tty34', '/dev/tty35', '/dev/tty36', '/dev/tty37', '/dev/tty38', '/dev/tty39', '/dev/tty40', '/dev/tty41', '/dev/tty42', '/dev/tty43', '/dev/tty44', '/dev/tty45', '/dev/tty46', '/dev/tty47', '/dev/tty48', '/dev/tty49', '/dev/tty50', '/dev/tty51', '/dev/tty52', '/dev/tty53', '/dev/tty54', '/dev/tty55', '/dev/tty56', '/dev/tty57', '/dev/tty58', '/dev/tty59', '/dev/tty60', '/dev/tty61', '/dev/tty62', '/dev/tty63', '/dev/ttyS0', '/dev/ttyS1', '/dev/ttyS2', '/dev/ttyS3', '/dev/ttyS4', '/dev/ttyS5', '/dev/ttyS6', '/dev/ttyS7', '/dev/ttyS8', '/dev/ttyS9', '/dev/ttyS10', '/dev/ttyS11', '/dev/ttyS12', '/dev/ttyS13', '/dev/ttyS14', '/dev/ttyS15', '/dev/ttyS16', '/dev/ttyS17', '/dev/ttyS18', '/dev/ttyS19', '/dev/ttyS20', '/dev/ttyS21', '/dev/ttyS22', '/dev/ttyS23', '/dev/ttyS24', '/dev/ttyS25', '/dev/ttyS26', '/dev/ttyS27', '/dev/ttyS28', '/dev/ttyS29', '/dev/ttyS30', '/dev/ttyS31', '/dev/ttyprintk', '/dev/ttyACM0', '/dev/ttyACM1']
Remove the USB cable from your MotorsBus and press Enter when done.

The port of this MotorsBus is '/dev/ttyACM1'
Reconnect the USB cable.
```
이렇게 각 모터드라이버에 연결된 USB의 이름을 확인하고 이 USB를 정상적으로 사용하기 위해 각 USB에 권한을 줘야합니다. 이 명령어의 경우 터미널을 닫거나 연결을 해제했을 때 항상 진행해줘야지 로봇을 정상적으로 사용할 수 있습니다.

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

만약 사용자의 USB 이름이 다르다면 /dev/ttyACM0 부분을 사용자에 맞춰서 변경해주면 됩니다.

이후의 설명은 공식 문서에 자세히 나와있어 공식문서를 참고해서 진행하시면 됩니다. 하드웨어 설치에 작성된 URL과 같으며 페이지의 하단 부분에 작성되어 있습니다.

[Hugging face Motor Configuration](https://huggingface.co/docs/lerobot/so101)

