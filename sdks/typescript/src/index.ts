export type ReliaGuardEvent = {
  projectId: string;
  userId: string;
  taskId: string;
  initialAnswer: string;
  initialConfidence: number;
  aiAdvice: string;
  aiConfidence: number;
  finalAnswer?: string;
  groundTruth?: string;
  context?: Record<string, unknown> | string;
  mode?: "audit" | "shadow" | "guardrail";
};

export type ReliaGuardAction = "allow" | "request_verification" | "show_uncertainty" | "delay" | "route_to_review";

export type GuardrailResponse = {
  state: string;
  risk: number;
  uncertainty: number;
  recommended_action: ReliaGuardAction;
  message: string;
  active_rules: string[];
  case_id: string;
  intervention_template: string;
  safety_boundary: string;
};

type ClientOptions = {
  apiKey?: string;
  baseUrl?: string;
};

function toSnakeCase(payload: ReliaGuardEvent): Record<string, unknown> {
  return {
    project_id: payload.projectId,
    user_id: payload.userId,
    task_id: payload.taskId,
    initial_answer: payload.initialAnswer,
    initial_confidence: payload.initialConfidence,
    ai_advice: payload.aiAdvice,
    ai_confidence: payload.aiConfidence,
    final_answer: payload.finalAnswer,
    ground_truth: payload.groundTruth,
    context: payload.context ?? {},
    mode: payload.mode ?? "shadow"
  };
}

export class ReliaGuard {
  private apiKey: string;
  private baseUrl: string;

  constructor(options: ClientOptions = {}) {
    this.apiKey = options.apiKey ?? "local-dev";
    this.baseUrl = options.baseUrl ?? "http://127.0.0.1:8000";
  }

  private async post<T>(path: string, body: Record<string, unknown>): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body)
    });
    if (!response.ok) {
      throw new Error(`ReliaGuard request failed: ${response.status} ${await response.text()}`);
    }
    return response.json() as Promise<T>;
  }

  async logInteraction(event: Required<Pick<ReliaGuardEvent, "finalAnswer" | "groundTruth">> & ReliaGuardEvent) {
    return this.post<{stored: boolean; case_id: string; guardrail: GuardrailResponse}>("/v1/events/log", toSnakeCase(event));
  }

  async checkGuardrail(event: ReliaGuardEvent) {
    return this.post<GuardrailResponse>("/v1/guardrail/check", toSnakeCase({...event, mode: event.mode ?? "guardrail"}));
  }
}
