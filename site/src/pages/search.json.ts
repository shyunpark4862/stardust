/**
 * The search index for the ⌘K palette: every API page (title, public names,
 * summary, place), every area overview and every Getting started page.
 * Built once with the site.
 */
import { getCollection } from 'astro:content';
import { areaOf, hrefOf, isSingle, splitId } from '../data/api';
import { plainText } from '../data/inline';
import { startHref } from '../data/start';

export async function GET() {
  const api = (await getCollection('api')).map((e) => {
    const [area] = splitId(e);
    return {
      t: e.data.title,
      n: e.data.names,
      s: plainText(e.data.summary),
      h: hrefOf(e),
      a: area,
      p: `${areaOf(area).name} · ${e.data.group}`,
      m: isSingle(e) ? 1 : 0,
    };
  });
  const areas = (await getCollection('areas')).map((e) => ({
    t: e.data.title,
    n: [e.data.ns],
    s: plainText(e.data.lead),
    h: `/api/${e.id}/`,
    a: e.id,
    p: 'API · Area',
    m: 0,
  }));
  const start = (await getCollection('start')).map((e) => ({
    t: e.data.title,
    n: [],
    s: plainText(e.data.summary),
    h: startHref(e),
    a: '',
    p: 'Get started',
    m: 0,
  }));
  return new Response(JSON.stringify([...api, ...areas, ...start]), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
