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
  { key: 'integrate', name: '적분', ns: 'sd.integrate', desc: 'QUADPACK 적응 구적, 겹친 적분, tanh-sinh, 다차원 cubature, 고정·표본 규칙', status: 'done' },
  { key: 'differentiate', name: '미분', ns: 'sd.differentiate', desc: '도함수, 야코비안, 헤시안의 수치 미분', status: 'done' },
  { key: 'random', name: '난수', ns: 'sd.random', desc: '계수기형 생성기, 스레드 수와 무관한 결정성', status: 'done' },
  { key: 'spatial', name: '공간', ns: 'sd.spatial', desc: '거리 행렬과 거리 함수 14종, 가장 가까운 이웃, KD 트리, Delaunay 삼각분할·볼록 껍질·보로노이 그림', status: 'done' },
  { key: 'fft', name: '푸리에 변환', ns: 'sd.fft', desc: '복소·실수·에르미트 FFT, DCT·DST 네 형, 다차원 변환, FFTLog', status: 'done' },
  { key: 'interpolate', name: '보간', ns: 'sd.interpolate', desc: '조각 다항식과 스플라인, 무게중심 다항식·유리식, 격자의 보간, 흩어진 점의 삼각분할·Clough–Tocher·RBF 보간', status: 'done' },
  { key: 'smooth', name: '평활', ns: 'sd.smooth', desc: 'lowess·loess, 평활 스플라인과 P-스플라인, 핵 회귀, 핵밀도추정과 대역폭, Savitzky–Golay', status: 'done' },
  { key: 'graph', name: '그래프', ns: 'sd.graph', desc: '순회와 연결 요소, 최단 경로, 신장 트리·흐름·매칭, 중심성', status: 'done' },
  { key: 'plot', name: '그림', ns: 'sd.plot', desc: '함수 그림, 논문용 통계 그림, 3D 곡면, PDF·SVG·PNG 출력', status: 'done' },
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
  { key: 'integrate', name: '적분', ns: 'sd.integrate', desc: 'QUADPACK의 Gauss–Kronrod와 외삽, 겹친 적분, 원소별 tanh-sinh와 급수, 벡터 값과 다차원 적분, 고정·표본 규칙, 정확도와 성능' },
  { key: 'differentiate', name: '미분', ns: 'sd.differentiate', desc: '유한 차분과 Richardson 외삽, 간격 고르기, 야코비안과 헤시안, 정확도와 성능' },
  { key: 'random', name: '난수', ns: 'sd.random', desc: '계수기형 생성기 Philox, 시드와 spawn, 정수·순열·가중치 추출의 알고리즘, 성능과 검증' },
  { key: 'spatial', name: '공간', ns: 'sd.spatial', desc: '타일 순서의 짝 커널, 유클리드 혼합 계산과 GEMM, KD 트리의 만들기와 가지치기, 이중 트리 순회, 단위 구 위의 haversine, 정확한 기하 술어와 기호 섭동, 나눠서 합치기와 하나씩 넣기의 삼각분할, 볼록 껍질과 보로노이, 정확도와 성능' },
  { key: 'fft', name: '푸리에 변환', ns: 'sd.fft', desc: '혼합 기수 Stockham과 Bluestein, 4단계 분해, 실수·에르미트 길, DCT·DST, FFTLog, 길이마다 한 경로' },
  { key: 'interpolate', name: '보간', ns: 'sd.interpolate', desc: '끝에서 정확한 선형 보간, 띠 풀이와 de Boor, 조각 다항식, 무게중심 꼴과 AAA, 격자와 RBF, 행렬식의 비로 구한 삼각분할 위의 무게와 Clough–Tocher' },
  { key: 'smooth', name: '평활', ns: 'sd.smooth', desc: '정렬된 창의 국소 직선, 여러 변수의 국소 다항식, 띠 Givens QR과 선택 역행렬, FITPACK 매듭, 가우스 합의 자르기와 구간화 FFT, 대역폭 찾기, Gram 다항식' },
  { key: 'graph', name: '그래프', ns: 'sd.graph', desc: '정의로 정한 결과와 동점 규칙, 최단 경로와 블록 Floyd–Warshall, 흐름과 매칭, PageRank와 Brandes' },
  { key: 'plot', name: '그림', ns: 'sd.plot', desc: '장면과 배치, LaTeX 글자와 글리프 캐시, 래스터라이저와 세 출력, 적응 표본, 3D 깊이 버퍼' },
] as const;
