import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

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

export default defineConfig({
  integrations: [mdx()],
  markdown: {
    // $...$ and $$...$$ render as math (KaTeX) in Markdown and MDX.
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: { theme: 'github-light', wrap: false, transformers: [replTransformer] },
  },
});
