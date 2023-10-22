const response = await fetch("https://api.runpod.ai/v2/k3b5bv3ncsey82/run", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${process.env.RUNPOD_KEY}`,
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
    `https://api.runpod.ai/v2/k3b5bv3ncsey82/stream/${id}`,
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

while (true) {
  const json = await stream();

  console.log(json.stream);

  if (json.status === "COMPLETED") {
    json.stream.forEach((x) => stream_response.push(x.output));
    break;
  }

  if (json.status === "IN_PROGRESS") {
    json.stream.forEach((x) => stream_response.push(x.output));
  }

  await new Promise((resolve) => setTimeout(resolve, 1000));
}

console.log(stream_response.map((x) => x).join(""));

export {};
