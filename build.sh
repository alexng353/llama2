VERSION = "0.0.1.5"

pnpm docker:build

docker tag llama2-serverless alexng353/llama2-runpod-serverless:$(VERSION)
docker push alexng353/llama2-runpod-serverless:$(VERSION)

