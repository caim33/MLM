ARG BASE_IMAGE=node:22-bookworm-slim
FROM ${BASE_IMAGE}
ARG NODE_VERSION=22.23.3
ARG NODE_SHA256=df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl openconnect vpnc-scripts iproute2 python3 xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN if ! command -v node >/dev/null; then \
    curl --fail --silent --show-error --location \
      "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz \
    && echo "${NODE_SHA256}  /tmp/node.tar.xz" | sha256sum --check - \
    && mkdir -p /opt/node && tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1 \
    && rm /tmp/node.tar.xz; fi
ENV PATH="/opt/node/bin:${PATH}"
WORKDIR /app
RUN npm install --omit=dev --ignore-scripts --save-exact sm-crypto@0.5.7
COPY aistation-cpu-keeper.mjs /app/aistation-cpu-keeper.mjs
COPY cpu-host/run.py /app/run.py
ENTRYPOINT ["python3", "/app/run.py"]
