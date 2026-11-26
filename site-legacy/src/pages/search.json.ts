/**
 * The search index for the ⌘K palette: every API page and article, with its
 * title, public names, summary and place. Built once with the site.
 */
import { getCollection } from 'astro:content';
import { areaOf, hrefOf, isSingle, splitId } from '../data/api';
import { internalsAreaOf, internalsHref, splitInternalsId } from '../data/internals';
import { plainText } from '../data/inline';

export async function GET() {
  const api = (await getCollection('api')).map((e) => {
    const [area] = splitId(e);
    return {
      t: e.data.title,
      n: e.data.names,
      s: plainText(e.data.summary),
      h: hrefOf(e),
      a: area,
      p: `API · ${areaOf(area).name} · ${e.data.group}`,
      m: isSingle(e) ? 1 : 0,
    };
  });
  const internals = (await getCollection('internals')).map((e) => {
    const [area] = splitInternalsId(e);
    return {
      t: e.data.title,
      n: [] as string[],
      s: plainText(e.data.summary),
      h: internalsHref(e),
      a: area,
      p: `구현 · ${internalsAreaOf(area).name} · ${e.data.group}`,
      m: 0,
    };
  });
  return new Response(JSON.stringify([...api, ...internals]), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
