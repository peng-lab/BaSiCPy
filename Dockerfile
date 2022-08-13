FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04

WORKDIR /
RUN apt-get update \
      && apt-get install -y libffi-dev gcc git curl python3-pip\
      && rm -rf /var/lib/apt/lists/*
#RUN git clone --branch v2.3.3 https://github.com/pyenv/pyenv.git /pyenv
#ENV PYENV_ROOT /pyenv
#RUN /pyenv/bin/pyenv install 3.9.13
#RUN eval "$(/pyenv/bin/pyenv init -)" && /pyenv/bin/pyenv local 3.9.13
RUN pip --no-cache-dir install --upgrade pip
RUN pip --no-cache-dir install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip --no-cache-dir install basicpy
# ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
# RUN python3 -c "import jax; print(jax.devices())"