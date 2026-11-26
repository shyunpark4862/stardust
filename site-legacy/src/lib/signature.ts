/**
 * Call signatures on a function page: wrap a long one over several lines and
 * turn each argument name into a link to its row in the reference card.
 */

/** Split "a, b=(1, 2), *, c" into top-level arguments. */
export function splitArgs(inner: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let quote = '';
  let cur = '';
  for (const ch of inner) {
    if (quote) {
      cur += ch;
      if (ch === quote) quote = '';
      continue;
    }
    if (ch === "'" || ch === '"') quote = ch;
    else if ('([{'.includes(ch)) depth++;
    else if (')]}'.includes(ch)) depth--;
    if (ch === ',' && depth === 0) {
      out.push(cur.trim());
      cur = '';
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

const escape = (s: string) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** Anchor id of a parameter row; a row may document several names ("x1, x2"). */
export const paramId = (name: string) => `param-${name.replace(/^\*+/, '')}`;

/** Map every documented name to the id of its row. */
export function paramIds(params: { name: string }[]): Map<string, string> {
  const ids = new Map<string, string>();
  for (const p of params) {
    const names = p.name.split(',').map((n) => n.trim().replace(/^\*+/, ''));
    for (const n of names) if (n && !ids.has(n)) ids.set(n, paramId(names[0]));
  }
  return ids;
}

/** One argument as HTML: the name links to its row when it is documented. */
function argHtml(arg: string, ids: Map<string, string>): string {
  if (arg === '/' || arg === '*' || arg === '...') return `<span class="sig__mark">${escape(arg)}</span>`;
  const m = arg.match(/^(\*{0,2})([A-Za-z_]\w*)(.*)$/s);
  if (!m) return escape(arg);
  const [, stars, name, rest] = m;
  const id = ids.get(name);
  const label = `${escape(stars)}${escape(name)}`;
  const head = id ? `<a class="sig__arg" href="#${id}">${label}</a>` : `<span class="sig__arg">${label}</span>`;
  return head + escape(rest);
}

/**
 * Render one signature. Short ones stay on one line; longer ones break after
 * the opening parenthesis and wrap the arguments greedily at `width` columns,
 * four spaces in.
 */
export function signatureHtml(
  sig: string,
  ids: Map<string, string>,
  ret?: string,
  width = 60,
): string {
  const open = sig.indexOf('(');
  const close = sig.lastIndexOf(')');
  const tail = ret ? ` <span class="sig__mark">→</span> <span class="sig__ret">${escape(ret)}</span>` : '';
  if (open < 0 || close < open) return escape(sig) + tail;
  const callee = sig.slice(0, open);
  const dot = callee.lastIndexOf('.');
  const calleeHtml =
    escape(callee.slice(0, dot + 1)) + `<span class="sig__name">${escape(callee.slice(dot + 1))}</span>`;
  const args = splitArgs(sig.slice(open + 1, close));
  const after = escape(sig.slice(close + 1));
  if (sig.length + (ret ? ret.length + 3 : 0) <= width + 8 || args.length < 2) {
    return `${calleeHtml}(${args.map((a) => argHtml(a, ids)).join(', ')})${after}${tail}`;
  }
  const indent = '    ';
  const lines: string[] = [];
  let line = '';
  let lineHtml = '';
  args.forEach((a, i) => {
    const piece = a + (i < args.length - 1 ? ',' : '');
    const pieceHtml = argHtml(a, ids) + (i < args.length - 1 ? ',' : '');
    if (line && indent.length + line.length + 1 + piece.length > width) {
      lines.push(indent + lineHtml);
      line = piece;
      lineHtml = pieceHtml;
    } else {
      line = line ? `${line} ${piece}` : piece;
      lineHtml = lineHtml ? `${lineHtml} ${pieceHtml}` : pieceHtml;
    }
  });
  if (lineHtml) lines.push(indent + lineHtml);
  return `${calleeHtml}(\n${lines.join('\n')}\n)${after}${tail}`;
}

/** A return type short enough to print after the arrow: one plain name. */
export const printableReturn = (type: string | undefined) =>
  type && /^[A-Za-z_][\w.]*(\[[\w., ]+\])?$/.test(type) ? type : undefined;
