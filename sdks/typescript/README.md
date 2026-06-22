# ReliaGuard TypeScript SDK

Install locally:

```bash
cd sdks/typescript
npm install
npm run build
```

Use in an AI app:

```ts
import { ReliaGuard } from "@reliaguard/studio";

const reliaguard = new ReliaGuard({
  apiKey: "local-dev",
  baseUrl: "http://127.0.0.1:8000"
});

await reliaguard.logInteraction({
  projectId: "ai-tutor",
  userId: "learner_42",
  taskId: "algebra_17",
  initialAnswer: "x = 4",
  initialConfidence: 0.61,
  aiAdvice: "x = 5",
  aiConfidence: 0.84,
  finalAnswer: "x = 5",
  groundTruth: "x = 4",
  context: { domain: "tutoring" }
});

const decision = await reliaguard.checkGuardrail({
  projectId: "ai-tutor",
  userId: "learner_42",
  taskId: "algebra_17",
  initialAnswer: "x = 4",
  initialConfidence: 0.61,
  aiAdvice: "x = 5",
  aiConfidence: 0.84,
  finalAnswer: "x = 5",
  groundTruth: "x = 4",
  context: { domain: "tutoring" }
});
```
