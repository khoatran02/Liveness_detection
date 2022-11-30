
# Start FROM Ubuntu image https://hub.docker.com/_/ubuntu
FROM ubuntu:20.04

RUN apt update && apt-get -y install sudo 

RUN useradd -m john
USER john
CMD /bin/bash
# Downloads to user config dir
# Install linux packages
# RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y python3-pip git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3.8-dev
# RUN alias python=python3

# Install pip packages
#COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install IPython mtcnn natsort opencv_python seaborn tensorflow

# Set up nivida docker

RUN curl https://get.docker.com | sh \
&&  systemctl --now enable docker

RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
             tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN  apt-get update
RUN  apt-get install -y nvidia-docker2

RUN  systemctl restart docker

RUN  docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi






# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . .

# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest-cpu && sudo docker build -f utils/docker/Dockerfile-cpu -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest-cpu && sudo docker pull $t && sudo docker run -it --ipc=host -v "$(pwd)"/datasets:/usr/src/datasets $t