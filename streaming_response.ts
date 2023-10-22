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

// parse streaming response

const reader = response.body?.getReader();

if (!reader) {
  throw new Error("No reader");
}

let result = await reader.read();

let text = "";

while (!result.done) {
  text += new TextDecoder().decode(result.value);
  result = await reader.read();
  // log without newline and flush
  process.stdout.write(`\r${text}`);
}

export {};
