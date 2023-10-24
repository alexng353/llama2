const RUNPOD_KEY = process.env.RUNPOD_KEY;
if (!RUNPOD_KEY) {
  throw new Error("No runpod api key provided");
}
const ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
if (!ENDPOINT_ID) {
  throw new Error("No endpoint id provided");
}

const response = await fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/run`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${RUNPOD_KEY}`,
  },
  body: JSON.stringify({
    input: {
      chat: [
        { role: "user", content: "Hello! What is your name?" },
        {
          role: "assistant",
          content:
            "Hello there, young mathematician! I'm Cosmo, your friendly and intelligent assistant here to help you learn all about limits and continuity in calculus!",
        },
        { role: "user", content: "What are limits?" },
      ],
    },
  }),
});

const json = await response.json();
console.log(json);

const id = json.id;

async function stream() {
  const streaming_response = await fetch(
    `https://api.runpod.ai/v2/${ENDPOINT_ID}/stream/${id}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.RUNPOD_KEY}`,
      },
    }
  );

  const json = await streaming_response.json();

  return json as { status: string; stream: { output: string }[] };
}

const stream_response: string[] = [];

let done = false;
while (!done) {
  const json = await stream();
  console.log(json.stream);

  switch (json.status) {
    case "COMPLETED":
      json.stream.forEach((x) => stream_response.push(x.output));
      done = true;
      break;
    case "IN_PROGRESS":
      json.stream.forEach((x) => stream_response.push(x.output));
      break;
  }

  await new Promise((resolve) => setTimeout(resolve, 1000));
}

console.log(stream_response.map((x) => x).join(""));

export {};
