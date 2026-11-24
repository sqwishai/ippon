// ippon in Pi: the decision models ippon serves as Pi classifiers, which codemode scripts (models.classify) and
// extensions (ctx.modelRegistry.classify) ask, and in an `ippon pi` session ippon's MCP server.
//
// `ippon pi` loads this file for one session and names the server (IPPON_URL, IPPON_MODEL) and the MCP server's
// command (IPPON_MCP). `ippon setup` links it into Pi's extensions for every session, where the MCP server is in Pi's
// mcp.json and the classifiers are the models of the server `ippon status` finds serving when Pi starts.
import { execFileSync } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const key = process.env.IPPON_API_KEY;

type Served = { url: string; models: { id: string; images: boolean }[] };

// the server's models, each with whether it reads images (catalogued builds report media before loading too)
async function served(): Promise<Served | undefined> {
  let url = process.env.IPPON_URL;
  try {
    if (!url) {
      const status = JSON.parse(execFileSync("ippon", ["status", "--json"], { encoding: "utf8", timeout: 15000 }));
      url = status.servers?.find((s: any) => s.status === "serving")?.url;
    }
    if (!url) return undefined;
    const auth = key ? { headers: { authorization: `Bearer ${key}` } } : {};
    const listed: any = await (await fetch(`${url}/v1/models`, auth)).json();
    const wanted = process.env.IPPON_MODEL;
    const models = listed.data
      .filter((m: any) => !wanted || m.id === wanted)
      .map((m: any) => ({ id: m.id, images: (m.media ?? []).includes("images") }));
    return { url, models };
  } catch {
    return undefined; // ippon not on PATH, or no server: Pi starts without ippon's classifiers
  }
}

// Pi's questions say `bool`, System One's say `noul`; answers go back in Pi's shapes
function classifier(url: string) {
  return async (target: any, context: any, options?: any) => {
    const out: any = {
      api: target.api,
      provider: target.provider,
      model: target.id,
      answers: {},
      stopReason: "stop",
      timestamp: Date.now(),
    };
    try {
      const images = (context.images ?? []).map((i: any) => `data:${i.mimeType};base64,${i.data}`);
      const questions = Object.fromEntries(
        Object.entries(context.questions).map(([id, q]: [string, any]) => [id, q.type === "bool" ? { ...q, type: "noul" } : q]),
      );
      const response = await fetch(`${url}/v1/systemone`, {
        method: "POST",
        headers: { "content-type": "application/json", ...(key ? { authorization: `Bearer ${key}` } : {}) },
        body: JSON.stringify({ model: target.id, state: context.state, questions, ...(images.length ? { images } : {}) }),
        signal: options?.signal,
      });
      const body: any = await response.json();
      if (!response.ok) throw new Error(body?.error?.message ?? `ippon answered ${response.status}`);
      for (const [id, q] of Object.entries(context.questions) as [string, any][]) {
        const a = body.answers[id];
        out.answers[id] =
          q.type === "bool"
            ? { type: "bool", probability: a.noul }
            : q.type === "choice"
              ? { type: "choice", choice: a.choice, probabilities: a.probabilities, confidence: a.confidence }
              : { type: "score", score: a.score, confidence: a.confidence };
      }
    } catch (error) {
      out.stopReason = options?.signal?.aborted ? "aborted" : "error";
      out.errorMessage = error instanceof Error ? error.message : String(error);
    }
    return out;
  };
}

export default async function (pi: ExtensionAPI) {
  const mcp = process.env.IPPON_MCP;
  if (mcp) {
    const [command, ...args] = JSON.parse(mcp);
    pi.registerMcpServer("ippon", { command, args });
  }
  const server = await served();
  if (!server) return;
  const cost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  pi.registerProvider("ippon", {
    name: "ippon",
    baseUrl: server.url,
    apiKey: key ?? "local",
    models: server.models.map(({ id, images }) => ({
      type: "classifier",
      id,
      name: id,
      api: "ippon-system-one",
      input: images ? ["text", "image"] : ["text"],
      cost,
      contextWindow: 32768,
    })),
    classifiers: { "ippon-system-one": { classify: classifier(server.url) } },
  } as any);
}
