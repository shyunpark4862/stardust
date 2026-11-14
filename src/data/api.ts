import { getCollection, type CollectionEntry } from 'astro:content';
import { areas, type AreaNav, type Crumb, type NavGroup } from './nav';

export type ApiEntry = CollectionEntry<'api'>;

/** "linalg/solve" -> ["linalg", "solve"] */
export function splitId(entry: ApiEntry): [area: string, slug: string] {
  const [area, slug] = entry.id.split('/');
  return [area, slug];
}

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
  linalg: ['선형 시스템', '분해'],
};

export function areaOf(key: string) {
  return areas.find((a) => a.key === key)!;
}

export function hrefOf(entry: ApiEntry): string {
  const [area, slug] = splitId(entry);
  return `/api/${area}/${slug}/`;
}

/** A page that documents exactly one function shows its name in the code font. */
export function isSingle(entry: ApiEntry): boolean {
  return entry.data.kind === 'function' && entry.data.names.length === 1;
}

/** An area's pages, grouped and ordered. */
export async function groupsOf(area: string): Promise<{ title: string; entries: ApiEntry[] }[]> {
  const entries = (await getCollection('api', (e) => e.id.startsWith(`${area}/`))).sort(
    (a, b) => a.data.order - b.data.order || a.data.title.localeCompare(b.data.title),
  );
  const groups = new Map<string, ApiEntry[]>();
  for (const entry of entries) {
    const list = groups.get(entry.data.group) ?? [];
    list.push(entry);
    groups.set(entry.data.group, list);
  }
  const rank = (title: string) => {
    const i = (groupOrder[area] ?? []).indexOf(title);
    return i === -1 ? Number.MAX_SAFE_INTEGER : i;
  };
  return [...groups]
    .map(([title, list]) => ({ title, entries: list }))
    .sort((a, b) => rank(a.title) - rank(b.title));
}

/** Areas that have at least one page. */
export async function documentedAreas(): Promise<string[]> {
  const entries = await getCollection('api');
  return [...new Set(entries.map((e) => splitId(e)[0]))];
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
export async function areaNav(area: string): Promise<AreaNav> {
  const documented = await documentedAreas();
  const info = areaOf(area);
  return {
    current: { key: info.key, name: info.name, ns: info.ns, href: `/api/${area}/` },
    all: areas.map((a) => ({
      key: a.key,
      name: a.name,
      ns: a.ns,
      href: documented.includes(a.key) ? `/api/${a.key}/` : undefined,
    })),
  };
}

/** Anchor of a group heading on its area page: "비교, 논리, 비트" -> "비교-논리-비트". */
export function groupId(title: string): string {
  return title.replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '');
}

/** API › area › group, for pages under /api/<area>/. */
export function areaCrumbs(area: string, group?: string): Crumb[] {
  const crumbs: Crumb[] = [
    { label: 'API', href: '/api/' },
    { label: areaOf(area).name, href: `/api/${area}/` },
  ];
  if (group) crumbs.push({ label: group, href: `/api/${area}/#${groupId(group)}` });
  return crumbs;
}
