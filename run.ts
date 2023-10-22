const response = await fetch("https://api.runpod.ai/v2/k3b5bv3ncsey82/run", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${process.env.RUNPOD_KEY}`,
  },
  body: JSON.stringify({
    input: {
      chat: [
        { role: "user", content: "Hello!" },
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

async function check() {
  const response = await fetch(
    `https://api.runpod.ai/v2/k3b5bv3ncsey82/status/${id}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.RUNPOD_KEY}`,
      },
    }
  );

  const json = await response.json();

  console.log(json);

  return json;
}

let counter = 0;
while (true) {
  console.log(`Checking for the ${counter++}th time...`);
  const json = await check();

  if (json.status === "COMPLETED") {
    break;
  }

  await new Promise((resolve) => setTimeout(resolve, 10_000));
}

export {};
