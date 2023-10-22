VERSION="0.0.1.6"

echo "Building version $VERSION"

pnpm docker:build

docker tag llama2-serverless alexng353/llama2-runpod-serverless:$VERSION
docker push alexng353/llama2-runpod-serverless:$VERSION

