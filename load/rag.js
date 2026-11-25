// k6 부하 시나리오 (2.11절). steady 상태에서 p95와 실패율을 봅니다.
import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

const latency = new Trend('rag_latency', true);

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: 20, timeUnit: '1s', duration: '5m',
      preAllocatedVUs: 40, maxVUs: 120,
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    rag_latency: ['p(95)<3500'],
  },
};

const questions = [
  '환불 처리 기한은?', '계약 갱신 조건 알려줘', 'API rate limit 정책은?',
  '장애 대응 절차 문서 있어?', '요금제별 지원 범위는?',
];

const TOKEN = __ENV.RAG_TOKEN || '';

export default function () {
  const q = questions[Math.floor(Math.random() * questions.length)];
  const res = http.post(`${__ENV.BASE || 'http://rag.localtest.me'}/v1/answer`,
    JSON.stringify({ question: q }),
    { headers: { 'content-type': 'application/json', authorization: `Bearer ${TOKEN}` } });
  latency.add(res.timings.duration);
  check(res, { 'status 200': (r) => r.status === 200 });
}
