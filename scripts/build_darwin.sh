#!/bin/sh

set -e

status() { echo >&2 ">>> $@"; }
usage() {
    echo "usage: $(basename $0) [build [sign]]"
    exit 1
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if [ -z "${VERSION:-}" ] || [ -z "${VERSION_SEMVER:-}" ] || [ -z "${VERSION_METADATA:-}" ]; then
    OLLAMA_ENV_QUIET=1 . "$SCRIPT_DIR/env.sh"
    unset OLLAMA_ENV_QUIET
else
    # Ensure helper variables are initialised when env.sh is not sourced.
    CARGO_FEATURES=${CARGO_FEATURES:-""}
fi
export VERSION
export VERSION_SEMVER=${VERSION_SEMVER:-${VERSION%%+*}}
export VERSION_METADATA=${VERSION_METADATA:-$VERSION}
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-11.3}

collect_runtime_libs() {
    target_triple=$1
    pattern=$2
    destination=$3

    runtime_dir="target/${target_triple}/release/runtime-libs"
    rm -rf "$runtime_dir"
    mkdir -p "$runtime_dir"

    for search_dir in "target/${target_triple}/release" "target/${target_triple}/release/deps"; do
        if [ -d "$search_dir" ]; then
            find "$search_dir" -maxdepth 1 -type f -name "$pattern" -exec cp {} "$runtime_dir"/ \;
        fi
    done

    if [ -n "$(ls -A "$runtime_dir" 2>/dev/null)" ]; then
        mkdir -p "$destination"
        for lib in "$runtime_dir"/*; do
            [ -f "$lib" ] || continue
            install -m755 "$lib" "$destination/$(basename "$lib")"
        done
    fi
}

ARCHS="arm64 amd64"
while getopts "a:h" OPTION; do
    case $OPTION in
        a) ARCHS=$OPTARG ;;
        h) usage ;;
    esac
done

shift $(( $OPTIND - 1 ))

_build_darwin() {
    for ARCH in $ARCHS; do
        status "Building darwin $ARCH"
        INSTALL_PREFIX=dist/darwin-$ARCH/
        mkdir -p "$INSTALL_PREFIX"

        case "$ARCH" in
            arm64) TARGET_TRIPLE=aarch64-apple-darwin ;;
            amd64) TARGET_TRIPLE=x86_64-apple-darwin ;;
            *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
        esac

        rustup target add "$TARGET_TRIPLE" >/dev/null 2>&1 || true
        if [ -n "$CARGO_FEATURES" ]; then
            cargo build --release --bin ollama --target "$TARGET_TRIPLE" --features "$CARGO_FEATURES"
        else
            cargo build --release --bin ollama --target "$TARGET_TRIPLE"
        fi
        install -Dm755 "target/$TARGET_TRIPLE/release/ollama" "$INSTALL_PREFIX/ollama"
        collect_runtime_libs "$TARGET_TRIPLE" "*.dylib" "$INSTALL_PREFIX/lib/ollama"

        if [ "$ARCH" = "amd64" ]; then
            status "Building darwin $ARCH dynamic backends"
            cmake -B build/darwin-$ARCH \
                -DCMAKE_OSX_ARCHITECTURES=x86_64 \
                -DCMAKE_OSX_DEPLOYMENT_TARGET=11.3 \
                -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
            cmake --build build/darwin-$ARCH --target ggml-cpu -j
            cmake --install build/darwin-$ARCH --component CPU
        fi
    done
}

_sign_darwin() {
    status "Creating universal binary..."
    mkdir -p dist/darwin
    lipo -create -output dist/darwin/ollama dist/darwin-*/ollama
    chmod +x dist/darwin/ollama

    if [ -n "$APPLE_IDENTITY" ]; then
        for F in dist/darwin/ollama dist/darwin-amd64/lib/ollama/*; do
            codesign -f --timestamp -s "$APPLE_IDENTITY" --identifier ai.ollama.ollama --options=runtime $F
        done

        # create a temporary zip for notarization
        TEMP=$(mktemp -u).zip
        ditto -c -k --keepParent dist/darwin/ollama "$TEMP"
        xcrun notarytool submit "$TEMP" --wait --timeout 10m --apple-id $APPLE_ID --password $APPLE_PASSWORD --team-id $APPLE_TEAM_ID
        rm -f "$TEMP"
    fi

    status "Creating universal tarball..."
    tar -cf dist/ollama-darwin.tar --strip-components 2 dist/darwin/ollama
    tar -rf dist/ollama-darwin.tar --strip-components 4 dist/darwin-amd64/lib/
    gzip -9vc <dist/ollama-darwin.tar >dist/ollama-darwin.tgz
}

_build_macapp() {
    # build and optionally sign the mac app
    npm install --prefix macapp
    if [ -n "$APPLE_IDENTITY" ]; then
        npm run --prefix macapp make:sign
    else
        npm run --prefix macapp make
    fi

    MACAPP_VERSION=${VERSION_SEMVER:-${VERSION%%+*}}
    MACAPP_ZIP="./macapp/out/make/zip/darwin/universal/Ollama-darwin-universal-${MACAPP_VERSION}.zip"
    if [ ! -f "$MACAPP_ZIP" ]; then
        MACAPP_ZIP="./macapp/out/make/zip/darwin/universal/Ollama-darwin-universal-${VERSION}.zip"
    fi
    if [ ! -f "$MACAPP_ZIP" ]; then
        echo "Unable to locate packaged app for version ${MACAPP_VERSION} or ${VERSION}" >&2
        exit 1
    fi
    mv "$MACAPP_ZIP" dist/Ollama-darwin.zip
}

if [ "$#" -eq 0 ]; then
    _build_darwin
    _sign_darwin
    _build_macapp
    exit 0
fi

for CMD in "$@"; do
    case $CMD in
        build) _build_darwin ;;
        sign) _sign_darwin ;;
        macapp) _build_macapp ;;
        *) usage ;;
    esac
done
