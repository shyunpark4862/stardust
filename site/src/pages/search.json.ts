/**
 * The search index for the ⌘K palette: every API page, with its
 * title, public names, summary and place. Built once with the site.
 */
import { getCollection } from 'astro:content';
import { areaOf, hrefOf, isSingle, splitId } from '../data/api';
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
  return new Response(JSON.stringify(api), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
