from pipeline.cloud.pipelines import run_pipeline
import webbrowser
MYSTIC_TOKEN = "pipeline_sk_yHbOiK9Glep_PDNXACovS5EieYaS3LsS"


output = run_pipeline(
    # Pipeline pointer or ID
    "stabilityai/stable-diffusion-xl-refiner-1.0:v1",
    # Prompt
    "Mountain winds and babbling springs and moonlight seas.",
    # Model kwargs
    dict(
        denoising_end=0.8,
        num_inference_steps=25,
    ),
)

result = output.result.result_array()

# Extract the image URL from the result
url = result[0][0]["file"]["url"]

# Open the URL in the default web browser
webbrowser.open(url)
