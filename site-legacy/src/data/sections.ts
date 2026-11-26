/**
 * What the API reference and the internals articles share: both are
 * collections of src/content/<section>/<area>/<slug>.mdx pages, listed by
 * area, grouped under headings and ordered inside each group.
 */
import { getCollection, type CollectionEntry } from 'astro:content';
import type { AreaNav, Crumb } from './nav';

export type Section = 'api' | 'internals';
export type SectionEntry<S extends Section> = CollectionEntry<S>;

/** The fields every page of both sections has. */
interface Placed {
  id: string;
  data: { title: string; group: string; order: number };
}

/** "linalg/solve" -> ["linalg", "solve"] */
export function splitEntryId(entry: { id: string }): [area: string, slug: string] {
  const [area, slug] = entry.id.split('/');
  return [area, slug];
}

/** `/<section>/<area>/<slug>/` */
export function entryHref(section: Section, entry: { id: string }): string {
  const [area, slug] = splitEntryId(entry);
  return `/${section}/${area}/${slug}/`;
}

/**
 * An area's pages, grouped. Groups follow `groupOrder` (unlisted groups come
 * last); pages inside a group follow `order`, then the title.
 */
export async function groupEntries<S extends Section>(
  section: S,
  area: string,
  groupOrder: string[] = [],
): Promise<{ title: string; entries: SectionEntry<S>[] }[]> {
  const entries = (await getCollection(section, (e: Placed) => e.id.startsWith(`${area}/`))) as (SectionEntry<S> & Placed)[];
  entries.sort((a, b) => a.data.order - b.data.order || a.data.title.localeCompare(b.data.title));
  const groups = new Map<string, SectionEntry<S>[]>();
  for (const entry of entries) {
    const list = groups.get(entry.data.group) ?? [];
    list.push(entry);
    groups.set(entry.data.group, list);
  }
  const rank = (title: string) => {
    const i = groupOrder.indexOf(title);
    return i === -1 ? Number.MAX_SAFE_INTEGER : i;
  };
  return [...groups]
    .map(([title, list]) => ({ title, entries: list }))
    .sort((a, b) => rank(a.title) - rank(b.title));
}

/** Areas of a section that have at least one page. */
export async function areasWithPages(section: Section): Promise<string[]> {
  const entries = await getCollection(section);
  return [...new Set(entries.map((e) => splitEntryId(e)[0]))];
}

/** Anchor of a group heading on its area page: "비교, 논리, 비트" -> "비교-논리-비트". */
export function groupId(title: string): string {
  return title.replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '');
}

interface AreaInfo {
  key: string;
  name: string;
  ns: string;
}

/** The area header of the sidebar: this area, and a way to the areas that have pages. */
export async function sectionAreaNav(section: Section, areas: readonly AreaInfo[], area: string): Promise<AreaNav> {
  const written = await areasWithPages(section);
  const info = areas.find((a) => a.key === area)!;
  return {
    current: { key: info.key, name: info.name, ns: info.ns, href: `/${section}/${area}/` },
    all: areas.map((a) => ({
      key: a.key,
      name: a.name,
      ns: a.ns,
      href: written.includes(a.key) ? `/${section}/${a.key}/` : undefined,
    })),
  };
}

/** Section › area › group. */
export function sectionCrumbs(section: Section, label: string, areaName: string, area: string, group?: string): Crumb[] {
  const crumbs: Crumb[] = [
    { label, href: `/${section}/` },
    { label: areaName, href: `/${section}/${area}/` },
  ];
  if (group) crumbs.push({ label: group, href: `/${section}/${area}/#${groupId(group)}` });
  return crumbs;
}
