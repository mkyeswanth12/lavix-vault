# Build stage
FROM node:22-alpine@sha256:16e22a550f3863206a3f701448c45f7912c6896a62de43add43bb9c86130c3e2 AS builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source files
COPY . .

# Build the frontend
RUN npm run build

# Post-build: rename JS files with real content hashes (Rolldown's [hash] is
# structurally stable — identical filenames for different source content).
RUN node hash-assets.cjs

# Production stage
FROM nginxinc/nginx-unprivileged:1.27-alpine@sha256:65e3e85dbaed8ba248841d9d58a899b6197106c23cb0ff1a132b7bfe0547e4c0

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --retries=10 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1

USER 101:101
CMD ["nginx", "-g", "daemon off;"]
