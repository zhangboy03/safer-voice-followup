import { tokenRequestBody } from "./callRouting.js";

function setQuery(search: string) {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      location: { search },
    },
  });
}

function parsedBody(lang?: "en" | "zh") {
  return JSON.parse(tokenRequestBody("room-1", "guest", lang)) as Record<
    string,
    unknown
  >;
}

setQuery("");
const englishBody = parsedBody("en");
if (englishBody.brain !== "solo")
  throw new Error("Public voice calls must pin Solo");
setQuery("?brain=brain&provider=openai");
if (parsedBody("en").brain !== "solo")
  throw new Error("Retired query flags must never select another voice engine");
