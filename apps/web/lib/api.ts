export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type RelianceCase = {
  initial_answer: string;
  initial_confidence: number;
  ai_advice: string;
  final_answer: string;
  ground_truth: string;
  task_context: string;
  advice_source: string;
  model_confidence: number;
};

export async function postRelianceCase(path: string, body: RelianceCase) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
    cache: "no-store"
  });
  if (!response.ok) throw new Error(`API ${path} failed`);
  return response.json();
}

export async function getJson(path: string) {
  const response = await fetch(`${API_BASE}${path}`, {cache: "no-store"});
  if (!response.ok) throw new Error(`API ${path} failed`);
  return response.json();
}
