# Common environment setup across build*.sh scripts

export VERSION=${VERSION:-$(git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")}
# TODO - consider `docker buildx ls --format=json` to autodiscover platform capability
PLATFORM=${PLATFORM:-"linux/arm64,linux/amd64"}
DOCKER_ORG=${DOCKER_ORG:-"ollama"}
FINAL_IMAGE_REPO=${FINAL_IMAGE_REPO:-"${DOCKER_ORG}/ollama"}
CARGO_FEATURES=${CARGO_FEATURES:-""}
RUST_TARGETS=${RUST_TARGETS:-""}
OLLAMA_COMMON_BUILD_ARGS="--build-arg=VERSION --build-arg=CARGO_FEATURES=${CARGO_FEATURES} --build-arg=RUST_TARGETS=${RUST_TARGETS}"

echo "Building Ollama"
echo "VERSION=$VERSION"
echo "PLATFORM=$PLATFORM"
if [ -n "$CARGO_FEATURES" ]; then
    echo "CARGO_FEATURES=$CARGO_FEATURES"
fi
if [ -n "$RUST_TARGETS" ]; then
    echo "RUST_TARGETS=$RUST_TARGETS"
fi
