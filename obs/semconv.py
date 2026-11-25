"""속성 이름의 단일 출처 (10.1절).

OpenTelemetry GenAI 시맨틱 컨벤션은 아직 바뀌는 중입니다(예: gen_ai.system →
gen_ai.provider.name). 이름은 이 파일에서만 정의하고, 다른 코드는 상수를 import합니다.
"""
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_PROVIDER = "gen_ai.provider.name"     # 구 gen_ai.system
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_CONVERSATION = "gen_ai.conversation.id"
RAG_TENANT = "rag.tenant"          # 해시된 값
RAG_GATE = "rag.gate"              # g1 | g2 | g3
RAG_GATE_VERDICT = "rag.gate.verdict"
RAG_GATE_P = "rag.gate.p"
RAG_TOP_K = "rag.retrieval.top_k"
RAG_HITS = "rag.retrieval.hits"
RAG_CITED = "rag.citation.cited"
RAG_SUPPORTED = "rag.citation.supported"
RAG_ABSTAINED = "rag.abstained"
