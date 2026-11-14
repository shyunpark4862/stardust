import { getCollection, type CollectionEntry } from 'astro:content';
import { internalsAreas, type AreaNav, type Crumb, type NavGroup } from './nav';
import { groupId } from './api';

export type InternalsEntry = CollectionEntry<'internals'>;

/** The order of the group headings on each area's page. */
const groupOrder: Record<string, string[]> = {
  core: ['전체 구조', '병렬 처리', 'Python과의 경계'],
  array: ['데이터', '연산 엔진', '집계와 통계', '인덱싱과 정렬', '행렬곱', '성능과 정확도'],
};

export function splitInternalsId(entry: InternalsEntry): [area: string, slug: string] {
  const [area, slug] = entry.id.split('/');
  return [area, slug];
}

export function internalsAreaOf(key: string) {
  return internalsAreas.find((a) => a.key === key)!;
}

export function internalsHref(entry: InternalsEntry): string {
  const [area, slug] = splitInternalsId(entry);
  return `/internals/${area}/${slug}/`;
}

export async function internalsGroups(area: string): Promise<{ title: string; entries: InternalsEntry[] }[]> {
  const entries = (await getCollection('internals', (e) => e.id.startsWith(`${area}/`))).sort(
    (a, b) => a.data.order - b.data.order || a.data.title.localeCompare(b.data.title),
  );
  const groups = new Map<string, InternalsEntry[]>();
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

export async function internalsAreasWithPages(): Promise<string[]> {
  const entries = await getCollection('internals');
  return [...new Set(entries.map((e) => splitInternalsId(e)[0]))];
}

export async function internalsSidebar(area: string): Promise<NavGroup[]> {
  return (await internalsGroups(area)).map((g) => ({
    title: g.title,
    items: g.entries.map((e) => ({ label: e.data.title, href: internalsHref(e) })),
  }));
}

export async function internalsAreaNav(area: string): Promise<AreaNav> {
  const written = await internalsAreasWithPages();
  const info = internalsAreaOf(area);
  return {
    current: { key: info.key, name: info.name, ns: info.ns, href: `/internals/${area}/` },
    all: internalsAreas.map((a) => ({
      key: a.key,
      name: a.name,
      ns: a.ns,
      href: written.includes(a.key) ? `/internals/${a.key}/` : undefined,
    })),
  };
}

export function internalsCrumbs(area: string, group?: string): Crumb[] {
  const crumbs: Crumb[] = [
    { label: '구현', href: '/internals/' },
    { label: internalsAreaOf(area).name, href: `/internals/${area}/` },
  ];
  if (group) crumbs.push({ label: group, href: `/internals/${area}/#${groupId(group)}` });
  return crumbs;
}
