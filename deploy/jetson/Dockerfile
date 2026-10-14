# Build on the target Jetson after selecting a JetPack-compatible NVIDIA PyTorch base.
# Example: docker build --build-arg BASE_IMAGE=<verified-image@sha256:digest> -f deploy/jetson/Dockerfile -t specific-range-edge .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /app
COPY requirements-edge.txt ./
RUN python3 -c "import sys,torch; assert sys.version_info >= (3,10); print(torch.__version__)"
RUN python3 -m pip install --no-cache-dir -r requirements-edge.txt
COPY src ./src
COPY scripts ./scripts
COPY docs ./docs
COPY README.md ./
ENV SPECIFIC_RANGE_NO_BROWSER=1
ENTRYPOINT ["python3", "scripts/run_edge_benchmark.py"]
