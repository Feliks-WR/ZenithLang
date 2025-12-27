FROM rust:1.89.0-slim

WORKDIR /app

# Copy only manifest first for better caching
COPY Cargo.toml ./
# Create a dummy main to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

# Now copy the real source
COPY src ./src

RUN cargo build --release

# We don't use ENTRYPOINT here so that CLion can pass its own arguments via 'cargo run'
CMD ["target/release/rust_parser"]