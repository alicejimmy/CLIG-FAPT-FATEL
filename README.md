# Unreliable Partial Label Learning: Novel Dataset Generation Method and Solution Frameworks
## Environment
我們在Ubuntu上運行及開發，以下是我們用到的套件版本：
conda create --name FAPT_FATEL python=3.11.4 ipykernel=6.25.0
conda activate FAPT_FATEL
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install wheel==0.41.2
pip install pandas==2.2.1
pip install Pillow==9.3.0
pip install numpy==1.24.1
你可以執行以下指令來安裝以上套件

或是你也可以直接使用CLIG-FAPT-FATEL/Dockerfile建立你的環境
### Packages

## 


docker build -t fapt_fatel_image -f Dockerfile/Dockerfile .

docker run \
--gpus '"device=0"' \
--name alicejimmy_final_test \
-it \
-v $HOME/ShareToContainer/:/home/ShareToContainer \
-p 15700:8888 \
-p 15701:22 \
-p 15702:3389 \
--shm-size="10g" \
fapt_fatel_image

docker start alicejimmy_final_test
docker exec -it alicejimmy_final_test /bin/bash
開始使用指令跑code

先用passwd修改其中root及user的密碼
/etc/init.d/ssh restart
ssh root@140.115.59.238 -p 15701
開始使用指令跑code
