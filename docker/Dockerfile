FROM cupy/cupy:latest

WORKDIR /work
RUN mkdir /bucket

ARG USER_ID
ARG USER_GROUP_ID

RUN addgroup --gid $USER_GROUP_ID ds
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $USER_GROUP_ID  ds

#Installing apt packages
RUN apt update
RUN apt install --yes vim git

#Installing python requirements
COPY requirements.pip ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.pip
RUN rm -f requirements.pip

#Setting a password (root) for the root user in case we needto use it with the image loaded.
RUN echo 'root:root' | chpasswd

USER ds
