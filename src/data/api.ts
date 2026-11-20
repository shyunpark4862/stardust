import type { CollectionEntry } from 'astro:content';
import { areas, type AreaNav, type Crumb, type NavGroup } from './nav';
import {
  areasWithPages,
  entryHref,
  groupEntries,
  groupId,
  sectionAreaNav,
  sectionCrumbs,
  splitEntryId,
} from './sections';

export { groupId };

export type ApiEntry = CollectionEntry<'api'>;

/** "linalg/solve" -> ["linalg", "solve"] */
export const splitId = (entry: ApiEntry) => splitEntryId(entry);

/** The order of the group headings on each area's page. */
const groupOrder: Record<string, string[]> = {
  array: [
    '기본 개념',
    '배열 만들기',
    '모양과 축',
    '잇기와 되풀이',
    '인덱싱 도구',
    '산술',
    '지수와 로그',
    '삼각함수와 쌍곡선함수',
    '반올림',
    '비교, 논리, 비트',
    '고르기와 판별',
    '복소수',
    '집계',
    'NaN을 건너뛰는 집계',
    '통계',
    '정렬, 탐색, 집합',
    '행렬곱',
    'dtype 도구',
    'ndarray 객체',
    '설정',
  ],
  linalg: ['기본 개념', '곱과 거듭제곱', '노름과 지표', '선형 시스템', '특수 구조', '분해', '행렬 함수'],
  frame: [
    '기본 개념',
    '만들기',
    '속성과 변환',
    '선택과 대입',
    '원소별 연산',
    '결측값',
    '리덕션',
    '중복과 고유값',
    '정렬',
    'groupby',
    '합치기',
  ],
  random: ['기본 개념', '생성기', '난수 뽑기'],
  stats: ['기본 개념', '연속 분포', '극값과 수명 분포', '비중심 분포', '이산 분포', '다변량 분포', '행렬 분포', '기술통계'],
  regression: ['기본 개념', '선형 회귀', '일반화 선형 모형', '정규화 회귀', '강건 회귀', '분위 회귀', '비선형 회귀'],
  optimize: ['기본 개념', '스칼라', '다변수 최소화', '최소제곱', '방정식계', '수치 미분'],
  special: [
    '기본 개념',
    '오차함수',
    '감마와 베타',
    '불완전 감마와 베타',
    '로지스틱과 정보량',
    '리덕션',
    '제타와 지수적분',
    'Bessel 함수',
    'Airy, 타원적분, 초기하',
    '복소수 함수',
    '분포 함수',
    '비중심 분포 함수',
  ],
};

export function areaOf(key: string) {
  return areas.find((a) => a.key === key)!;
}

export function hrefOf(entry: ApiEntry): string {
  return entryHref('api', entry);
}

/**
 * A page that documents exactly one function shows its name in the code font.
 * A method page (e.g. rng.random) lists no public names of its own but is
 * titled by the method name, so it counts too, and so does a page for one
 * method that several classes share (astype: Series.astype, DataFrame.astype).
 */
export function isSingle(entry: ApiEntry): boolean {
  const { kind, names, title } = entry.data;
  if (kind !== 'function') return false;
  if (names.length === 1) return true;
  if (!/^[A-Za-z_][\w.]*$/.test(title)) return false;
  return names.every((n) => n === title || n.endsWith(`.${title}`));
}

/** An area's pages, grouped and ordered. */
export function groupsOf(area: string): Promise<{ title: string; entries: ApiEntry[] }[]> {
  return groupEntries('api', area, groupOrder[area]);
}

/** Areas that have at least one page. */
export function documentedAreas(): Promise<string[]> {
  return areasWithPages('api');
}

/** Sidebar groups for every page under /api/<area>/. */
export async function areaSidebar(area: string): Promise<NavGroup[]> {
  const groups = await groupsOf(area);
  return groups.map((g) => ({
    title: g.title,
    items: g.entries.map((e) => ({
      label: e.data.title,
      href: hrefOf(e),
      mono: isSingle(e),
      names: isSingle(e) ? undefined : e.data.names,
    })),
  }));
}

/** The area header of the sidebar: this area, and a way to the others. */
export function areaNav(area: string): Promise<AreaNav> {
  return sectionAreaNav('api', areas, area);
}

/** API › area › group, for pages under /api/<area>/. */
export function areaCrumbs(area: string, group?: string): Crumb[] {
  return sectionCrumbs('api', 'API', areaOf(area).name, area, group);
}
