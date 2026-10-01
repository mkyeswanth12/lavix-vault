#!/bin/sh
# Lavix Vault MinIO init (runs as the one-shot `minio-init` service).
# Creates the private bucket and reconciles the restricted application
# user/policy on every controlled startup. Reads all four credentials
# from the environment. POSIX sh.
set -eu

test -n "$MINIO_ROOT_USER" && test -n "$MINIO_ROOT_PASSWORD"
test -n "$MINIO_APP_ACCESS_KEY" && test -n "$MINIO_APP_SECRET_KEY"
# Root and app identities must stay separate (preflight also checks).
test "$MINIO_ROOT_USER" != "$MINIO_APP_ACCESS_KEY"
test "$MINIO_ROOT_PASSWORD" != "$MINIO_APP_SECRET_KEY"
bucket="$MINIO_BUCKET_NAME"
case "$bucket" in
  ""|*[!a-z0-9.-]*|.*|*.|-*|*-|*..*|*.-*|*-.*)
    printf 'Invalid MINIO_BUCKET_NAME: %s\n' "$bucket" >&2
    exit 1
    ;;
esac
test "${#bucket}" -ge 3
test "${#bucket}" -le 63
mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mb --ignore-existing "local/$bucket"
mc anonymous set private "local/$bucket"
printf '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:GetBucketLocation","s3:ListBucket"],"Resource":["arn:aws:s3:::%s"]},{"Effect":"Allow","Action":["s3:DeleteObject","s3:GetObject","s3:PutObject"],"Resource":["arn:aws:s3:::%s/*"]}]}\n' \
  "$bucket" "$bucket" > /tmp/lavix-bucket-policy.json
mc admin user add local "$MINIO_APP_ACCESS_KEY" "$MINIO_APP_SECRET_KEY"
mc admin policy create local lavix-bucket-access /tmp/lavix-bucket-policy.json
mc admin policy attach local lavix-bucket-access --user "$MINIO_APP_ACCESS_KEY"
