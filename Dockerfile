FROM alpine/git:latest AS builder

# Fetch model source code
RUN mkdir -p /opt && cd /opt && git clone git://github.com/cpllab/gpt-2.git gpt-2 && \
        cd gpt-2 && git checkout lm-zoo && rm -rf .git

# Build arguments provide SSH keys for accessing private CPL data.
ARG CPL_SSH_PRV_KEY
RUN mkdir /root/.ssh && echo "StrictHostKeyChecking no" >> /root/.ssh/config \
        && echo "$CPL_SSH_PRV_KEY" > /root/.ssh/id_rsa \
        && chmod 600 /root/.ssh/id_rsa

# Download model hyperparameters without checkpoint
RUN mkdir -p /opt/gpt-2/model && cd /opt/gpt-2/model && \
        curl -sO https://storage.googleapis.com/gpt-2/models/117M/{encoder.json,hparams.json,vocab.bpe}

# Copy in model parameters.
ARG CHECKPOINT_NAME=bllip-lg-gptbpe_1581955288
RUN scp -o "StrictHostKeyChecking=no" \
        cpl@45.79.223.150:/home/cpl/gpt-2/models/${CHECKPOINT_NAME}/{checkpoint,model.\*} /opt/gpt-2/model

# Remove SSH information.
RUN rm -rf /root/.ssh

FROM tensorflow/tensorflow:1.12.0-gpu-py3

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=8.0" \
    LANG=C.UTF-8

# Add runtime dependencies
RUN pip3 install regex tqdm fire requests toposort

COPY --from=builder /opt/gpt-2 /opt/gpt-2
