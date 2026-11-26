/**
 * Frontmatter text (summaries, parameter descriptions) is plain YAML, so
 * Markdown in it is not rendered by MDX. These helpers give it the two pieces
 * of inline Markdown the reference uses there: `code` and [links](/path/).
 */
import { url } from '../lib/url';

const escape = (s: string) =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

const codeHtml = (text: string) =>
  text
    .split(/(`[^`]+`)/)
    .map((part) => (part.startsWith('`') && part.endsWith('`') && part.length > 1 ? `<code>${escape(part.slice(1, -1))}</code>` : escape(part)))
    .join('');

/** `code` spans become <code>, [text](/path/) becomes a link (with the site base); the rest is escaped. */
export function inlineHtml(text: string): string {
  return text
    .split(/(\[[^\]]+\]\([^)\s]+\))/)
    .map((part) => {
      const m = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/);
      return m ? `<a href="${escape(url(m[2]))}">${codeHtml(m[1])}</a>` : codeHtml(part);
    })
    .join('');
}

/** The same text with the Markdown dropped, for <meta>, search and other plain places. */
export function plainText(text: string): string {
  return text.replace(/\[([^\]]+)\]\([^)\s]+\)/g, '$1').replace(/`([^`]+)`/g, '$1');
}
