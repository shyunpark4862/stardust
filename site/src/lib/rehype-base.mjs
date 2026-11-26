/**
 * Links in Markdown and MDX are written from the site root ("](/api/linalg/)").
 * When the site is built under a sub-path (`base` in astro.config.mjs), this
 * puts the base in front of every root-relative href and src, so the pages
 * themselves never need to know where the site is served.
 *
 * Left alone: #anchors, relative paths, other hosts ("//host", "https:"), and
 * the props of MDX components (capitalized tags), which apply the base
 * themselves through src/lib/url.ts.
 */

const ATTRS = new Set(['href', 'src']);

/** @param {{ base?: string }} options */
export function rehypeBase({ base = '/' } = {}) {
  const prefix = base.replace(/\/$/, '');
  const fix = (v) => (typeof v === 'string' && v.startsWith('/') && !v.startsWith('//') ? prefix + v : v);
  const walk = (node) => {
    if (node.type === 'element' && node.properties) {
      for (const k of ATTRS) if (k in node.properties) node.properties[k] = fix(node.properties[k]);
    } else if ((node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') && /^[a-z]/.test(node.name ?? '')) {
      for (const a of node.attributes ?? []) if (a.type === 'mdxJsxAttribute' && ATTRS.has(a.name)) a.value = fix(a.value);
    }
    for (const c of node.children ?? []) walk(c);
  };
  // At the root there is nothing to add: no transformer.
  return prefix ? (tree) => walk(tree) : undefined;
}
