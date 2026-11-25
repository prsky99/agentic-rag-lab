"""서비스 계층. 판단은 하네스에 있고, 서비스는 HTTP·인증·직렬화만 담당합니다."""
# 관측성 초기화(10.1절): 어떤 서비스든 `services.<이름>.main`을 import하는 순간 이 파일이
# 먼저 실행됩니다. OTEL_EXPORTER_OTLP_ENDPOINT가 있고 OTEL_SDK_DISABLED가 아니면
# Provider·Exporter를 설정하고, FastAPI 앱마다 추적 미들웨어를 붙이며, 나가는 요청에
# traceparent를 넣습니다. 조건이 맞지 않으면 아무것도 하지 않습니다.
from services.common.otel import autoinstrument as _autoinstrument

_autoinstrument()
