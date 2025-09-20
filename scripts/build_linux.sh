#!/bin/sh
#
# Mac ARM users, rosetta can be flaky, so to use a remote x86 builder
#
# docker context create amd64 --docker host=ssh://mybuildhost
# docker buildx create --name mybuilder amd64 --platform linux/amd64
# docker buildx create --name mybuilder --append desktop-linux --platform linux/arm64
# docker buildx use mybuilder


set -eu

. $(dirname $0)/env.sh

mkdir -p dist

docker buildx build \
        --output type=local,dest=./dist/ \
        --platform=${PLATFORM} \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --target archive \
        -f Dockerfile \
        .

if echo $PLATFORM | grep "amd64" > /dev/null; then
    outDir="./dist"
    if echo $PLATFORM | grep "," > /dev/null ; then
       outDir="./dist/linux_amd64"
    fi
    docker buildx build \
        --output type=local,dest=${outDir} \
        --platform=linux/amd64 \
        ${OLLAMA_COMMON_BUILD_ARGS} \
        --build-arg FLAVOR=rocm \
        --target archive \
        -f Dockerfile \
        .
fi

# buildx behavior changes for single vs. multiplatform

compress_bundle() {
    src_dir="$1"
    shift
    output="$1"
    shift
    tar c -C "$src_dir" "$@" | pigz -9vc >"$output"
}

create_archives_for_arch() {
    root="$1"
    arch="$2"

    echo "Creating bundle archives for linux/${arch} from ${root}"

    base_archive="./dist/ollama-linux-${arch}.tar.gz"
    compress_bundle "$root" "$base_archive" \
        --exclude "lib/ollama/cuda_jetpack5" \
        --exclude "lib/ollama/cuda_jetpack6" \
        --exclude "lib/ollama/rocm" \
        bin lib

    if [ -d "$root/lib/ollama/cuda_jetpack5" ]; then
        compress_bundle "$root" "./dist/ollama-linux-${arch}-gpu-jetpack5.tar.gz" \
            lib/ollama/cuda_jetpack5
    fi

    if [ -d "$root/lib/ollama/cuda_jetpack6" ]; then
        compress_bundle "$root" "./dist/ollama-linux-${arch}-gpu-jetpack6.tar.gz" \
            lib/ollama/cuda_jetpack6
    fi

    if [ -d "$root/lib/ollama/rocm" ]; then
        compress_bundle "$root" "./dist/ollama-linux-${arch}-gpu-rocm.tar.gz" \
            lib/ollama/rocm
    fi
}

echo "Compressing linux tar bundles..."
if echo $PLATFORM | grep "," > /dev/null ; then
    create_archives_for_arch "./dist/linux_arm64" "arm64"
    create_archives_for_arch "./dist/linux_amd64" "amd64"
elif echo $PLATFORM | grep "arm64" > /dev/null ; then
    create_archives_for_arch "./dist" "arm64"
elif echo $PLATFORM | grep "amd64" > /dev/null ; then
    create_archives_for_arch "./dist" "amd64"
fi
