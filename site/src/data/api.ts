/**
 * The API reference: pages in src/content/api/<area>/<slug>.mdx, area
 * overviews in src/content/areas/<area>.mdx. Group order and group
 * descriptions come from the overview's frontmatter; a page whose group the
 * overview does not list stops the build.
 */
import { getCollection, type CollectionEntry } from 'astro:content';
import { areas, type AreaNav, type Crumb, type NavGroup } from './nav';

export type ApiEntry = CollectionEntry<'api'>;
export type AreaEntry = CollectionEntry<'areas'>;

/** "linalg/solve" -> ["linalg", "solve"] */
export function splitId(entry: { id: string }): [area: string, slug: string] {
  const [area, slug] = entry.id.split('/');
  return [area, slug];
}

export function hrefOf(entry: { id: string }): string {
  const [area, slug] = splitId(entry);
  return `/api/${area}/${slug}/`;
}

/** The area's entry in the site's list of areas (name, namespace, card text). */
export function areaOf(key: string) {
  const info = areas.find((a) => a.key === key);
  if (!info) throw new Error(`unknown area "${key}" (src/data/nav.ts lists the areas)`);
  return info;
}

/** Anchor of a group heading on its area page: "Linear systems" -> "linear-systems". */
export function groupId(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-|-$/g, '');
}

/** The overview entry of an area, or undefined when the area has none yet. */
export async function overviewOf(area: string): Promise<AreaEntry | undefined> {
  return (await getCollection('areas')).find((e) => e.id === area);
}

/** Areas that have an overview page, in the order of src/data/nav.ts. */
export async function documentedAreas(): Promise<string[]> {
  const ids = new Set((await getCollection('areas')).map((e) => e.id));
  for (const id of ids) areaOf(id);
  return areas.map((a) => a.key).filter((k) => ids.has(k));
}

export interface Group {
  title: string;
  desc: string;
  entries: ApiEntry[];
}

/**
 * An area's pages under the overview's groups, in the overview's order;
 * inside a group by `order`, then title. Fails the build when a page's area
 * has no overview or its group is not one of the overview's groups.
 */
export async function groupsOf(area: string): Promise<Group[]> {
  const overview = await overviewOf(area);
  const entries = await getCollection('api', (e) => e.id.startsWith(`${area}/`));
  if (!overview) {
    if (entries.length === 0) return [];
    throw new Error(`api/${entries[0].id}.mdx: area "${area}" has no overview (src/content/areas/${area}.mdx)`);
  }
  const groups: Group[] = overview.data.groups.map((g) => ({ ...g, entries: [] }));
  for (const e of entries) {
    const g = groups.find((x) => x.title === e.data.group);
    if (!g) {
      const known = overview.data.groups.map((x) => `"${x.title}"`).join(', ');
      throw new Error(`api/${e.id}.mdx: group "${e.data.group}" is not in src/content/areas/${area}.mdx groups (${known})`);
    }
    g.entries.push(e);
  }
  for (const g of groups) g.entries.sort((a, b) => a.data.order - b.data.order || a.data.title.localeCompare(b.data.title));
  return groups;
}

/** All pages of an area in sidebar order. */
export async function orderedEntries(area: string): Promise<ApiEntry[]> {
  return (await groupsOf(area)).flatMap((g) => g.entries);
}

/**
 * The page a reference ("solve", "linalg/solve") points at, seen from `area`,
 * or undefined when that page does not exist (yet).
 */
export async function resolveRef(area: string, ref: string): Promise<ApiEntry | undefined> {
  const id = ref.includes('/') ? ref : `${area}/${ref}`;
  return (await getCollection('api')).find((e) => e.id === id);
}

/**
 * A page that documents exactly one name shows its title in the code font,
 * and so does a method page titled by the method name.
 */
export function isSingle(entry: ApiEntry): boolean {
  const { kind, names, title } = entry.data;
  if (kind === 'concept') return false;
  if (!/^[A-Za-z_][\w.]*$/.test(title)) return false;
  // A class page lists its members (Class.member); a method page shared by
  // several classes lists Series.astype, DataFrame.astype ...
  return names.length <= 1 || names.every((n) => n === title || n.endsWith(`.${title}`) || n.startsWith(`${title}.`));
}

/** Sidebar groups for every page under /api/<area>/. Empty groups are left out. */
export async function areaSidebar(area: string): Promise<NavGroup[]> {
  return (await groupsOf(area))
    .filter((g) => g.entries.length > 0)
    .map((g) => ({
      title: g.title,
      items: g.entries.map((e) => ({
        label: e.data.title,
        href: hrefOf(e),
        mono: isSingle(e),
        names: isSingle(e) || e.data.names.length === 0 ? undefined : e.data.names,
      })),
    }));
}

/** The area header of the sidebar: this area, and the way to the others. */
export async function areaNav(area: string): Promise<AreaNav> {
  const written = await documentedAreas();
  const info = areaOf(area);
  return {
    current: { key: info.key, name: info.name, ns: info.ns, href: `/api/${area}/` },
    all: areas.map((a) => ({
      key: a.key,
      name: a.name,
      ns: a.ns,
      href: written.includes(a.key) ? `/api/${a.key}/` : undefined,
    })),
  };
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

/**
 * Where a component placed in an MDX body (<Reference />, <FunctionTables />)
 * sits among the body's headings: the number of h2/h3 headings before it,
 * fenced code skipped. The table of contents inserts the component's own
 * headings at that point.
 */
export function headingsBefore(body: string | undefined, tag: string): number | undefined {
  if (!body) return undefined;
  let count = 0;
  let fence = false;
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (fence) continue;
    if (new RegExp(`^\\s*<${tag}\\b`).test(line)) return count;
    if (/^#{2,3}\s/.test(line)) count++;
  }
  return undefined;
}
