/**
 * Frontmatter text (summaries, descriptions) is plain YAML, so Markdown in it
 * is not rendered by MDX. These helpers give it the one piece of inline
 * Markdown the docs use there: `code`.
 */

const escape = (s: string) =>
  s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');

/** `code` spans become <code>; everything else is escaped text. */
export function inlineHtml(text: string): string {
  return text
    .split(/(`[^`]+`)/)
    .map((part) => (part.startsWith('`') && part.endsWith('`') && part.length > 1 ? `<code>${escape(part.slice(1, -1))}</code>` : escape(part)))
    .join('');
}

/** The same text with the backticks dropped, for <meta> and other plain places. */
export function plainText(text: string): string {
  return text.replace(/`([^`]+)`/g, '$1');
}
