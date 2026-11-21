import type { CollectionEntry } from 'astro:content';
import { internalsAreas, type AreaNav, type Crumb, type NavGroup } from './nav';
import { areasWithPages, entryHref, groupEntries, sectionAreaNav, sectionCrumbs, splitEntryId } from './sections';

export type InternalsEntry = CollectionEntry<'internals'>;

/** The order of the group headings on each area's page. */
const groupOrder: Record<string, string[]> = {
  core: ['전체 구조', '병렬 처리', 'Python과의 경계', '검증과 측정'],
  array: ['데이터', '연산 엔진', '집계와 통계', '인덱싱과 정렬', '행렬곱', '성능과 정확도'],
  linalg: ['구조', '기본 연산', '선형 시스템과 분해', '고유값과 특이값', '특수 구조와 행렬 함수', '성능과 정확도'],
  frame: ['구조', '연산', '입출력', '병렬화와 성능'],
  random: ['생성기', '뽑기', '성능과 검증'],
  special: ['구조', '수치 기법', '함수별 알고리즘', '정확도와 성능'],
  stats: ['구조', '분포별 알고리즘', '표본과 다변량', '기술통계', '정확도와 성능'],
  regression: ['구조', '선형 모형', '일반화 선형 모형', '정규화', '강건과 분위', '비선형', '검정과 진단', '변수 선택', '정확도와 성능'],
  sparse: ['구조', '형식과 연산', '직접 풀이', '반복법', '고유값과 특이값', '행렬 함수', '정확도와 성능'],
  htest: ['구조', '모수 검정', '적합도', '순위와 표본 비교', '상관', '분할표와 비율', '다중 비교', '재표본', '정확도와 성능'],
  integrate: ['구조', 'QUADPACK', '겹친 적분과 벡터 값', 'tanh-sinh와 급수', '다차원', '고정·표본 규칙', '정확도와 성능'],
  differentiate: ['구조', '알고리즘', '정확도와 성능'],
  optimize: ['구조', '스칼라', '다변수 최소화', '최소제곱과 방정식계', '정확도와 성능'],
};

export const splitInternalsId = (entry: InternalsEntry) => splitEntryId(entry);

export function internalsAreaOf(key: string) {
  return internalsAreas.find((a) => a.key === key)!;
}

export function internalsHref(entry: InternalsEntry): string {
  return entryHref('internals', entry);
}

export function internalsGroups(area: string): Promise<{ title: string; entries: InternalsEntry[] }[]> {
  return groupEntries('internals', area, groupOrder[area]);
}

export function internalsAreasWithPages(): Promise<string[]> {
  return areasWithPages('internals');
}

export async function internalsSidebar(area: string): Promise<NavGroup[]> {
  return (await internalsGroups(area)).map((g) => ({
    title: g.title,
    items: g.entries.map((e) => ({ label: e.data.title, href: internalsHref(e) })),
  }));
}

export function internalsAreaNav(area: string): Promise<AreaNav> {
  return sectionAreaNav('internals', internalsAreas, area);
}

export function internalsCrumbs(area: string, group?: string): Crumb[] {
  return sectionCrumbs('internals', '구현', internalsAreaOf(area).name, area, group);
}
