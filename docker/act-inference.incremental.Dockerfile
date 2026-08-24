ARG BASE_RUNTIME_IMAGE=excavator-act-inference:base-de6f83f
FROM ${BASE_RUNTIME_IMAGE}

ARG EXCAVATOR_IL_REVISION=unknown

COPY docker/wheelhouse/pyserial-3.5-py2.py3-none-any.whl /tmp/
RUN python3 -m pip install --no-cache-dir --no-index \
    /tmp/pyserial-3.5-py2.py3-none-any.whl

WORKDIR /opt/excavator-il
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation . && \
    python3 -c "import excavator_il.resident_act_runtime, lerobot, serial, torch; assert '.nv26.01.' in torch.__version__"

LABEL org.opencontainers.image.revision="${EXCAVATOR_IL_REVISION}"
