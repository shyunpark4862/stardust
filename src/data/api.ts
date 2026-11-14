import { getCollection, type CollectionEntry } from 'astro:content';
import { areas, type NavGroup } from './nav';

export type ApiEntry = CollectionEntry<'api'>;

/** "linalg/solve" -> ["linalg", "solve"] */
export function splitId(entry: ApiEntry): [area: string, slug: string] {
  const [area, slug] = entry.id.split('/');
  return [area, slug];
}

/** The order of the group headings on each area's page. */
const groupOrder: Record<string, string[]> = {
  linalg: ['선형 시스템', '분해'],
};

export function areaOf(key: string) {
  return areas.find((a) => a.key === key)!;
}

export function hrefOf(entry: ApiEntry): string {
  const [area, slug] = splitId(entry);
  return `/api/${area}/${slug}/`;
}

/** An area's functions, grouped in the order the groups first appear. */
export async function groupsOf(area: string): Promise<{ title: string; entries: ApiEntry[] }[]> {
  const entries = (await getCollection('api', (e) => e.id.startsWith(`${area}/`))).sort(
    (a, b) => a.data.order - b.data.order || a.data.name.localeCompare(b.data.name),
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

/** Areas that have at least one documented function. */
export async function documentedAreas(): Promise<string[]> {
  const entries = await getCollection('api');
  return [...new Set(entries.map((e) => splitId(e)[0]))];
}

/** Sidebar for every page under /api/<area>/. */
export async function areaSidebar(area: string): Promise<NavGroup[]> {
  const groups = await groupsOf(area);
  return [
    {
      title: 'API',
      items: [
        { label: '전체 영역', href: '/api/' },
        { label: `${areaOf(area).name} 개요`, href: `/api/${area}/` },
      ],
    },
    ...groups.map((g) => ({
      title: g.title,
      items: g.entries.map((e) => ({ label: e.data.name, href: hrefOf(e), mono: true })),
    })),
  ];
}
