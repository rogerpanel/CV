# CICFlowMeter container: the SAME extractor family as the ICS3D container flows,
# so testbed features line up with the saved models' schema. Pinned so every run
# of the study uses one build (the plan requires one CICFlowMeter build per run).
FROM eclipse-temurin:11-jdk AS build
ARG CFM_REF=master
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# CICFlowMeter (Java) builds a CLI that turns a pcap into flow CSV
RUN git clone https://github.com/ahlashkari/CICFlowMeter.git . && git checkout "$CFM_REF" || true
RUN ./gradlew --no-daemon fatJar || ./gradlew --no-daemon jar || true

FROM eclipse-temurin:11-jre
RUN apt-get update && apt-get install -y --no-install-recommends libpcap0.8 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build /opt/cfm/build
COPY extract.sh /opt/cfm/extract.sh
RUN chmod +x /opt/cfm/extract.sh
WORKDIR /opt/cfm
ENTRYPOINT ["/opt/cfm/extract.sh"]
