FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils nano git curl python3-pip dirmngr gnupg2 && \
    rm -rf /var/lib/apt/lists/*

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

#  Install Python dependencies
COPY requirements.txt ssl_requirements.txt
COPY bayesian_erfnet/requirements.txt dl_requirements.txt

RUN pip3 install -r dl_requirements.txt && \
    pip3 install -r ssl_requirements.txt && \
    rm ssl_requirements.txt && \
    rm dl_requirements.txt && \
    rm -r ~/.cache/pip
