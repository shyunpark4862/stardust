/** Getting started pages: src/content/start/<slug>.mdx, listed by `order`. */
import { getCollection, type CollectionEntry } from 'astro:content';
import type { NavGroup } from './nav';

export type StartEntry = CollectionEntry<'start'>;

export async function startPages(): Promise<StartEntry[]> {
  return (await getCollection('start')).sort((a, b) => a.data.order - b.data.order || a.data.title.localeCompare(b.data.title));
}

export const startHref = (e: { id: string }) => `/start/${e.id}/`;

/** The sidebar of every page under /start/. */
export async function startSidebar(): Promise<NavGroup[]> {
  return [
    {
      title: 'Get started',
      items: [{ label: 'Overview', href: '/start/' }, ...(await startPages()).map((e) => ({ label: e.data.title, href: startHref(e) }))],
    },
  ];
}
