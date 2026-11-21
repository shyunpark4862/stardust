export interface NavItem {
  label: string;
  href: string;
  badge?: string;
  /** Show the label in the code font (function names). */
  mono?: boolean;
  /** Public names on a family page: shown under the label and matched by the filter. */
  names?: string[];
}
export interface NavGroup {
  title: string;
  items: NavItem[];
}

/** One step of the path shown above a page title. */
export interface Crumb {
  label: string;
  href?: string;
}

/** The area a reference page belongs to, and where the other areas are. */
export interface AreaNav {
  current: { key: string; name: string; ns: string; href: string };
  all: { key: string; name: string; ns: string; href?: string }[];
}

/** Top navigation. One entry per site section. */
export const topNav: NavItem[] = [
  { label: '소개', href: '/intro/' },
  { label: 'API', href: '/api/' },
  { label: '구현', href: '/internals/' },
];

/** Sidebar trees, keyed by site section. */
export const sidebars: Record<string, NavGroup[]> = {
  intro: [
    {
      title: '소개',
      items: [{ label: 'stardust란', href: '/intro/' }],
    },
  ],
  api: [
    {
      title: 'API',
      items: [{ label: '개요', href: '/api/' }],
    },
    {
      title: '영역',
      items: [{ label: '선형대수', href: '/api/linalg/' }],
    },
  ],
  internals: [
    {
      title: '구현',
      items: [{ label: '개요', href: '/internals/' }],
    },
  ],
};

export const areas = [
  { key: 'array', name: '배열', ns: 'sd', desc: 'N차원 ndarray, ufunc, 리덕션, 인덱싱, 정렬', status: 'done' },
  { key: 'linalg', name: '선형대수', ns: 'sd.linalg', desc: '분해, 선형 시스템, 노름, 특수 구조, expm', status: 'done' },
  { key: 'sparse', name: '희소 행렬', ns: 'sd.sparse', desc: 'COO·CSR·CSC·DOK 형식, 희소 연산, 직접·반복 풀이, 고유값, 행렬 함수', status: 'done' },
  { key: 'frame', name: '데이터프레임', ns: 'sd.frame', desc: 'Series, DataFrame, NA, groupby, merge, read_csv', status: 'done' },
  { key: 'special', name: '특수함수', ns: 'sd.special', desc: '오차함수, 감마, 불완전 감마·베타, Bessel, 초기하', status: 'done' },
  { key: 'stats', name: '통계', ns: 'sd.stats', desc: '단변량·다변량·행렬 분포, 기술통계', status: 'done' },
  { key: 'htest', name: '가설검정', ns: 'sd.htest', desc: 't 검정과 분산분석, 적합도, 순위, 상관, 분할표, 비율, 다중 비교, 재표본', status: 'done' },
  { key: 'regression', name: '회귀분석', ns: 'sd.regression', desc: '선형 모형, GLM, 정규화, 강건·분위 회귀, 비선형 최소제곱', status: 'done' },
  { key: 'optimize', name: '최적화', ns: 'sd.optimize', desc: '스칼라 근과 최솟값, 다변수 최소화, 최소제곱, 방정식계, 수치 미분', status: 'done' },
  { key: 'random', name: '난수', ns: 'sd.random', desc: '계수기형 생성기, 스레드 수와 무관한 결정성', status: 'done' },
] as const;

/** Areas of the internals section. "core" is what every area is built on. */
export const internalsAreas = [
  { key: 'core', name: '공통 기반', ns: 'stardust', desc: '층 구조와 빌드, 스레드풀과 결정성, Python과 C++의 경계' },
  { key: 'array', name: '배열', ns: 'sd', desc: '데이터 모델, 연산 엔진, 커널, 집계, 인덱싱, 정렬, 행렬곱, 성능과 정확도' },
  { key: 'linalg', name: '선형대수', ns: 'sd.linalg', desc: 'LAPACK을 감싸는 구조, 패킹과 배치, LU·QR·고유값·SVD의 알고리즘, 행렬 지수, 성능과 정확도' },
  { key: 'sparse', name: '희소 행렬', ns: 'sd.sparse', desc: '형식과 표준 꼴, 희소 행렬곱, Accelerate 희소 분해, Krylov 반복법, 고유값과 특이값, 행렬 함수, 정확도와 성능' },
  { key: 'frame', name: '데이터프레임', ns: 'sd.frame', desc: '컬럼과 Copy-on-Write, Swiss table과 factorize, 기수 정렬과 groupby, merge, CSV 파싱, 병렬화와 성능' },
  { key: 'special', name: '특수함수', ns: 'sd.special', desc: '원소별 엔진, 유리 근사·연분수·점근전개 같은 수치 기법, 함수별 알고리즘, 정확도와 성능' },
  { key: 'stats', name: '통계', ns: 'sd.stats', desc: '분포 엔진과 빠른 경로, 연속·이산·비중심 분포의 수치 기법, 표본, 다변량 밀도와 누적확률, 기술통계, 정확도와 성능' },
  { key: 'htest', name: '가설검정', ns: 'sd.htest', desc: '검정의 공통 구조, 정확한 분포와 근사, 순위와 동점, 분할표의 정확 검정, 다중 비교, 재표본, 정확도와 성능' },
  { key: 'regression', name: '회귀분석', ns: 'sd.regression', desc: '최소제곱과 반복 개선, 추론과 진단, IRLS와 분리 판정, 좌표 하강과 KKT 연마, 강건·분위 회귀, 비선형 최소제곱, 정확도와 성능' },
  { key: 'optimize', name: '최적화', ns: 'sd.optimize', desc: '사용자 함수의 경계와 수치 미분, 구간법과 Chandrupatla, Nelder–Mead·BFGS·L-BFGS-B·신뢰 영역, Levenberg–Marquardt와 TRF, Powell 혼성법, 정확도와 성능' },
  { key: 'random', name: '난수', ns: 'sd.random', desc: '계수기형 생성기 Philox, 시드와 spawn, 정수·순열·가중치 추출의 알고리즘, 성능과 검증' },
] as const;
