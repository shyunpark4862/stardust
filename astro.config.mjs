import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { rehypeHeadingIds } from '@astrojs/markdown-remark';

/** Removes the first `count` characters of a highlighted line. */
function stripPrefix(line, count) {
  for (const span of line.children) {
    if (count <= 0) break;
    for (const text of span.children ?? []) {
      if (text.type !== 'text' || count <= 0) continue;
      const cut = Math.min(count, text.value.length);
      text.value = text.value.slice(cut);
      count -= cut;
    }
  }
}

/**
 * Interactive-session examples (`>>> ` input, bare output). The prompts are
 * drawn by CSS, so selecting the code copies only what one would type, and
 * output lines are marked so they can be set apart from input.
 */
const replTransformer = {
  name: 'repl',
  pre(node) {
    if (this.source.includes('>>> ')) this.addClassToHast(node, 'repl');
  },
  line(node, number) {
    if (!this.source.includes('>>> ')) return;
    const text = this.source.split('\n')[number - 1] ?? '';
    const input = text.startsWith('>>> ') || text === '>>>';
    const more = text.startsWith('... ') || text === '...';
    if (input || more) {
      stripPrefix(node, 4);
      this.addClassToHast(node, input ? 'repl-in' : 'repl-more');
    } else {
      this.addClassToHast(node, 'repl-out');
    }
  },
};

/**
 * Headings that contain math: records, per heading id, the heading as a list
 * of text and TeX pieces (frontmatter `headingParts`), so the table of
 * contents can draw the formula instead of its source. It runs after the
 * heading ids exist and before KaTeX replaces the math nodes; from then on a
 * heading's plain text would hold every formula three times (MathML, TeX
 * annotation, HTML).
 */
function rehypeHeadingMath() {
  const isMath = (n) => n.type === 'element' && (n.properties?.className ?? []).includes('language-math');
  const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));
  const partsOf = (n, out) => {
    for (const c of n.children ?? []) {
      if (isMath(c)) out.push({ tex: textOf(c) });
      else if (c.type === 'text') out.push({ text: c.value });
      else partsOf(c, out);
    }
    return out;
  };
  const walk = (n, found) => {
    if (n.type === 'element' && /^h[2-4]$/.test(n.tagName) && n.properties?.id) {
      const parts = partsOf(n, []);
      if (parts.some((p) => p.tex !== undefined)) found[n.properties.id] = parts;
      return;
    }
    for (const c of n.children ?? []) walk(c, found);
  };
  return (tree, file) => {
    const found = {};
    walk(tree, found);
    if (Object.keys(found).length > 0) file.data.astro.frontmatter.headingParts = found;
  };
}

export default defineConfig({
  integrations: [mdx()],
  markdown: {
    // GFM is added by hand so a single tilde stays a tilde: Korean text writes
    // ranges as "5~27%", and two of them in a line must not become strikethrough
    // (~~text~~ still does).
    gfm: false,
    // $...$ and $$...$$ render as math (KaTeX) in Markdown and MDX.
    remarkPlugins: [[remarkGfm, { singleTilde: false }], remarkMath],
    // Ids first, so a heading's id and text come from its TeX source, not
    // from KaTeX's output.
    rehypePlugins: [rehypeHeadingIds, rehypeHeadingMath, rehypeKatex],
    shikiConfig: { theme: 'github-light', wrap: false, transformers: [replTransformer] },
  },
});
