#!/bin/bash
docker build --platform linux/amd64 -t asia-northeast3-docker.pkg.dev/eventstorming-tool-db/eventstorming-repo/project-generator:latest . --no-cache
docker push asia-northeast3-docker.pkg.dev/eventstorming-tool-db/eventstorming-repo/project-generator:latest
