// 결정 모델 다운 시나리오 (11.9절).
//   kubectl -n rag scale deploy/decision --replicas=0
//   RAG_TOKEN=$(cat /tmp/rag-token) k6 run load/decision-down.js
// k6는 check가 실패해도 그것만으로는 실패 종료하지 않습니다. thresholds의 checks 조건이
// 깨져야 종료 코드가 0이 아니게 됩니다. 그래서 checks 통과율 100%를 임계값으로 둡니다.
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 20, duration: '3m',
  thresholds: { checks: ['rate==1.0'] },
};

// 200이면 기권 본문(abstained·reason)까지, 503이면 JSON 사유까지 확인합니다(tests/verify.sh와 같은 기준).
const REASON = __ENV.EXPECT_REASON || 'g3_unavailable';

function body(r) {
  try { return r.json(); } catch (e) { return null; }
}

export default function () {
  const res = http.post(`${__ENV.BASE || 'http://rag.localtest.me'}/v1/answer`,
    JSON.stringify({ question: '환불 규정' }),
    { headers: { 'content-type': 'application/json',
                 authorization: `Bearer ${__ENV.RAG_TOKEN || ''}` } });
  const b = body(res);
  check(res, {
    'degraded but alive': (r) => r.status === 200 || r.status === 503,
    'abstained with reason': (r) => b !== null && (
      (r.status === 200 && b.abstained === true && b.status === 'abstained' && b.reason === REASON)
      || (r.status === 503 && ['g3_unavailable', 'decision_unavailable']
        .includes(String(b.reason || b.error || '')))),
  });
}
