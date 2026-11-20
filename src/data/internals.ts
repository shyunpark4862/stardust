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
  regression: ['구조', '선형 모형', '일반화 선형 모형', '정규화', '강건과 분위', '비선형', '정확도와 성능'],
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
