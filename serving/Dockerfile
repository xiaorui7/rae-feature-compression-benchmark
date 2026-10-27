FROM golang:1.26.2-bookworm AS build
WORKDIR /src
COPY serving/go.mod serving/go.sum ./
RUN go mod download
COPY serving/ ./
RUN mkdir /out && CGO_ENABLED=0 go build -o /out/ ./cmd/...

FROM scratch
COPY --from=build /out/ /bin/
USER 65532:65532
CMD ["/bin/shard", "--listen=0.0.0.0:9090", "--admin=0.0.0.0:9091"]
