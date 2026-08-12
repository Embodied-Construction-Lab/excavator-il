FROM nvcr.io/nvidia/pytorch:26.01-py3-igpu@sha256:1a7c59f29e84393a8de02413bed2cfc80f3aa70389b35165bc52b0fbe4cbb29e

ARG LEROBOT_COMMIT=12b88fce029cc3a8a94b061cd9e790018873c769
ARG EXCAVATOR_IL_REVISION=unknown

# The field Wi-Fi currently has a very slow IPv6 route to PyPI. Keep this
# preference inside the build image; do not change the Orin host network.
RUN printf 'precedence ::ffff:0:0/96  100\n' >> /etc/gai.conf

COPY docker/act-inference.requirements.txt /tmp/act-inference.requirements.txt
COPY docker/wheelhouse /tmp/wheelhouse
ARG PIP_OFFLINE=0
RUN if [ "${PIP_OFFLINE}" = 1 ]; then \
      python3 -m pip install --no-cache-dir --no-index \
        --find-links=/tmp/wheelhouse \
        -r /tmp/act-inference.requirements.txt && \
      python3 -m pip install --no-cache-dir --no-index --no-deps \
        --no-build-isolation \
        "/tmp/wheelhouse/lerobot-${LEROBOT_COMMIT}.tar.gz"; \
    else \
      python3 -m pip install --no-cache-dir \
        -r /tmp/act-inference.requirements.txt && \
      python3 -m pip install --no-cache-dir --no-deps \
        "lerobot @ git+https://github.com/freshmakerzhao/lerobot.git@${LEROBOT_COMMIT}"; \
    fi

WORKDIR /opt/excavator-il
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation . && \
    python3 -c "import excavator_il, lerobot, torch; assert '.nv26.01.' in torch.__version__"

LABEL org.opencontainers.image.title="excavator ACT inference"
LABEL org.opencontainers.image.description="Offline ACT checkpoint validation on Jetson Orin"
LABEL org.opencontainers.image.revision="${EXCAVATOR_IL_REVISION}"
