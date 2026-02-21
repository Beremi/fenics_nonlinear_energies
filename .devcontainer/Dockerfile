FROM dolfinx/lab:stable

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1

RUN set -eux; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
    ffmpeg \
    git \
    gcc \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --upgrade \
    ipympl \
    ipywidgets \
    opencv-python \
    jax \
    jaxlib \
    pyamg \
    igraph \
    scalene \
    Jinja2 \
    meshio
