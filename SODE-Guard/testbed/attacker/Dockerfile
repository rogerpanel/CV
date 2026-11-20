# Attacker image: standard, public security tools only. No custom exploit code.
# Used inside the internal-only cluster to generate labelled attack traffic
# against the testbed's own vulnerable containers.
FROM debian:12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      nmap hydra hping3 iproute2 curl ca-certificates python3 python3-pip \
      git golang-go iptables procps \
 && rm -rf /var/lib/apt/lists/*
# nuclei (vulnerability scanner) from upstream release, pinned
ARG NUCLEI_VERSION=3.3.5
RUN curl -fsSL -o /tmp/nuclei.zip \
      https://github.com/projectdiscovery/nuclei/releases/download/v${NUCLEI_VERSION}/nuclei_${NUCLEI_VERSION}_linux_amd64.zip \
 && (cd /usr/local/bin && unzip -o /tmp/nuclei.zip nuclei) || true
# slowloris (public PoC) for the slow-DoS scenario
RUN pip3 install --no-cache-dir --break-system-packages slowloris==0.1.4 requests==2.31.0
COPY *.py targets.env /opt/attacker/
WORKDIR /opt/attacker
