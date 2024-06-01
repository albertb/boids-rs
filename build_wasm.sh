#!/bin/sh

cargo build --release --target wasm32-unknown-unknown && \
wasm-bindgen --no-typescript --target web \
--out-dir ./docs --out-name boids \
./target/wasm32-unknown-unknown/release/boids.wasm
