// 단일 요청 스모크. 배포 직후 직접 실행합니다.
import http from 'k6/http';
import { check } from 'k6';

export const options = { vus: 1, iterations: 3 };

export default function () {
  const res = http.post(`${__ENV.BASE || 'http://rag.localtest.me'}/v1/answer`,
    JSON.stringify({ question: '환불은 며칠 이내에 가능한가요?' }),
    { headers: { 'content-type': 'application/json',
                 authorization: `Bearer ${__ENV.RAG_TOKEN || ''}` } });
  check(res, { 'status 200': (r) => r.status === 200 });
}
