ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    FC=gfortran-15 CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=/usr/bin/g++-14
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
COPY containers/install.sh /tmp/install.sh
RUN bash /tmp/install.sh
# Source-only context: results, environments and build products are excluded.
COPY . /workspace
RUN mkdir -p /output && chmod 1777 /output
ENTRYPOINT ["/workspace/containers/entrypoint.sh"]
CMD ["doctor"]
