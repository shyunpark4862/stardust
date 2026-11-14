export interface NavItem {
  label: string;
  href: string;
  badge?: string;
}
export interface NavGroup {
  title: string;
  items: NavItem[];
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
      items: [
        { label: '선형대수', href: '/api/linalg/', badge: '샘플' },
      ],
    },
  ],
  internals: [
    {
      title: '구현',
      items: [{ label: '개요', href: '/internals/' }],
    },
    {
      title: '주제',
      items: [
        { label: '스레드풀', href: '/internals/thread-pool/', badge: '샘플' },
      ],
    },
  ],
};

export const areas = [
  { key: 'array', name: '배열', ns: 'sd', desc: 'N차원 ndarray, ufunc, 리덕션, 인덱싱, 정렬', status: 'done' },
  { key: 'linalg', name: '선형대수', ns: 'sd.linalg', desc: '분해, 선형 시스템, 노름, 특수 구조, expm', status: 'done' },
  { key: 'frame', name: '데이터프레임', ns: 'sd.frame', desc: 'Series, DataFrame, NA, groupby, merge, read_csv', status: 'done' },
  { key: 'special', name: '특수함수', ns: 'sd.special', desc: '오차함수, 감마, 불완전 감마·베타, Bessel, 초기하', status: 'done' },
  { key: 'stats', name: '통계', ns: 'sd.stats', desc: '단변량·다변량 분포, 기술통계, 검정', status: 'draft' },
  { key: 'random', name: '난수', ns: 'sd.random', desc: '계수기형 생성기, 스레드 수와 무관한 결정성', status: 'draft' },
] as const;
