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

/**
 * A Korean subject particle that fits the word: 이 after a final consonant,
 * 가 after a vowel ("배열이", "특수함수가"). Words that end in something other
 * than a Hangul syllable take the vowel form.
 */
export function subject(word: string): string {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  const final = code >= 0 && code < 11172 && code % 28 !== 0;
  return `${word}${final ? '이' : '가'}`;
}
