/**
 * TeX → SVG at build time, with MathJax 4 and the STIX Two math font.
 *
 * Every formula on the site goes through here: the rehype plugin below for
 * the page bodies, and `renderTex` for the table of contents. Nothing runs
 * in the browser.
 */
import { mathjax } from '@mathjax/src/mjs/mathjax.js';
import { TeX } from '@mathjax/src/mjs/input/tex.js';
import { SVG } from '@mathjax/src/mjs/output/svg.js';
import { liteAdaptor } from '@mathjax/src/mjs/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from '@mathjax/src/mjs/handlers/html.js';
import '@mathjax/src/mjs/util/asyncLoad/esm.js';
import '@mathjax/src/mjs/input/tex/base/BaseConfiguration.js';
import '@mathjax/src/mjs/input/tex/ams/AmsConfiguration.js';
import '@mathjax/src/mjs/input/tex/newcommand/NewcommandConfiguration.js';
import '@mathjax/src/mjs/input/tex/boldsymbol/BoldsymbolConfiguration.js';
import '@mathjax/src/mjs/input/tex/textmacros/TextMacrosConfiguration.js';
import '@mathjax/src/mjs/input/tex/mathtools/MathtoolsConfiguration.js';
import '@mathjax/src/mjs/input/tex/cancel/CancelConfiguration.js';
import '@mathjax/src/mjs/input/tex/color/ColorConfiguration.js';
import { MathJaxStix2Font } from '@mathjax/mathjax-stix2-font/mjs/svg.js';

// The adaptor cannot measure text, so text the font has no glyph for (in
// \text{...}) is sized by estimate and drawn in the body font. The CJK width
// only matters for wide characters, which the English text does not use.
const adaptor = liteAdaptor({ cjkCharWidth: 0.95, unknownCharWidth: 0.5 });
RegisterHTMLHandler(adaptor);

const tex = new TeX({
  packages: ['base', 'ams', 'newcommand', 'boldsymbol', 'textmacros', 'mathtools', 'cancel', 'color'],
  formatError: (_jax, err) => {
    throw err;
  },
});
const output = (fontCache) =>
  new SVG({
    fontData: MathJaxStix2Font,
    fontCache,
    dynamicPrefix: '@mathjax/mathjax-stix2-font/mjs/svg/dynamic',
    mtextInheritFont: false,
  });
// Page bodies: one glyph cache per page. Each glyph's path is written once,
// in a hidden <svg> at the end of the page, and the formulas <use> it; that
// halves the size of a page with many formulas.
const pageSvg = output('global');
const pageDoc = mathjax.document('', { InputJax: tex, OutputJax: pageSvg });
// Anything else (the table of contents): self-contained SVG.
const svg = output('none');
const doc = mathjax.document('', { InputJax: tex, OutputJax: svg });

// One page or one formula at a time: MathJax keeps state while it loads font
// ranges on demand, and the page cache belongs to the page being rendered.
let queue = Promise.resolve();
const exclusive = (fn) => {
  const run = queue.then(fn);
  queue = run.catch(() => {});
  return run;
};
const convert = (source, display) => exclusive(() => doc.convertPromise(source, { display }));

/** A formula as an HTML string (the <mjx-container> with its SVG). */
export async function renderTex(source, display = false) {
  return adaptor.outerHTML(stripLatex(await convert(source, display)));
}

function stripLatex(node) {
  if (node.attributes) for (const k of Object.keys(node.attributes)) if (k.startsWith('data-latex')) delete node.attributes[k];
  for (const c of node.children ?? []) stripLatex(c);
  return node;
}

/** The stylesheet the SVG output needs (containers, line breaks). */
export function mathCss() {
  return adaptor.cssText(svg.styleSheet(doc));
}

/** A lite DOM node → hast. */
function toHast(node) {
  // Inline math is split into pieces at the points where a line may break,
  // with <mjx-break> holding a space between them. MDX drops text that is
  // only whitespace, so the space is a zero-width space here (still a break
  // opportunity); prose.css gives the break its width.
  if (node.kind === 'mjx-break') {
    return { type: 'element', tagName: 'mjx-break', properties: { ...node.attributes }, children: [{ type: 'text', value: '\u200B' }] };
  }
  if (node.kind === '#text') return { type: 'text', value: node.value };
  if (node.kind === '#comment') return { type: 'comment', value: node.value };
  const properties = {};
  for (const [k, v] of Object.entries(node.attributes ?? {})) {
    // The TeX of every sub-expression rides along as data-latex; the page
    // has no use for it, and it would double the size of each formula.
    if (!k.startsWith('data-latex')) properties[k] = v;
  }
  return { type: 'element', tagName: node.kind, properties, children: (node.children ?? []).map(toHast) };
}

const classes = (n) => n.properties?.className ?? [];
const isMath = (n) => n.type === 'element' && classes(n).includes('language-math');
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));

/**
 * rehype plugin: replaces the math nodes of remark-math
 * (`<code class="language-math math-inline">`, and
 * `<pre><code class="language-math math-display">`) with MathJax SVG.
 * Inline math becomes <span class="math math-inline">, display math
 * <div class="math math-display">. A formula MathJax cannot read stops the
 * build with the file and the TeX.
 */
export function rehypeMathjax() {
  return async (tree, file) => {
    const jobs = [];
    const walk = (node, parent, index) => {
      if (node.type === 'element' && node.tagName === 'pre' && node.children?.length === 1 && isMath(node.children[0])) {
        jobs.push({ parent, index, source: textOf(node.children[0]), display: true });
        return;
      }
      // A paragraph that is nothing but one formula is a display formula,
      // also when it was written on one line ($$...$$ in running text reads
      // as inline math).
      if (node.type === 'element' && node.tagName === 'p') {
        const kids = (node.children ?? []).filter((c) => !(c.type === 'text' && c.value.trim() === ''));
        if (kids.length === 1 && isMath(kids[0])) {
          jobs.push({ parent, index, source: textOf(kids[0]), display: true });
          return;
        }
      }
      if (isMath(node)) {
        jobs.push({ parent, index, source: textOf(node), display: classes(node).includes('math-display') });
        return;
      }
      (node.children ?? []).forEach((c, i) => walk(c, node, i));
    };
    walk(tree, null, 0);
    if (jobs.length === 0) return;
    await exclusive(async () => {
      pageSvg.fontCache.clearCache();
      for (const job of jobs) {
        let out;
        try {
          out = await pageDoc.convertPromise(job.source, { display: job.display });
        } catch (err) {
          throw new Error(`MathJax: ${err.message}\n  in ${file.path}\n  ${job.source}`);
        }
        job.parent.children[job.index] = {
          type: 'element',
          tagName: job.display ? 'div' : 'span',
          properties: { className: ['math', job.display ? 'math-display' : 'math-inline'], 'aria-label': job.source.trim(), role: 'math' },
          children: [toHast(out)],
        };
      }
      // The glyph paths, once, in a hidden <svg> that every formula's <use>
      // points into. It takes no room (display: none hides the drawing, not
      // the definitions).
      tree.children.push({
        type: 'element',
        tagName: 'svg',
        properties: { xmlns: 'http://www.w3.org/2000/svg', style: 'display: none', 'aria-hidden': 'true', className: ['math-glyphs'] },
        children: [toHast(pageSvg.fontCache.getCache())],
      });
    });
  };
}
