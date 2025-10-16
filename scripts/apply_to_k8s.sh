#!/bin/bash
kubectl delete deployment.apps/project-generator
kubectl apply -f k8s/deployment.yaml
