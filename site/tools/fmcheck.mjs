// Frontmatter check for the site's pages.
//
//   node tools/fmcheck.mjs [files ...]      (from site/; no files: every page)
//
// Checks what the build's schema (src/content.config.ts) does not, against
// the pinned build's inventory (notes/api-inventory.json):
// - YAML parses; required fields are there; no unknown fields.
// - api pages: `group` is one of the area overview's groups; every `names`
//   entry is a public name (Class.member for members); every `signatures`
//   entry is `sd.<area>.<name>(...)` with exactly the pinned signature (a
//   method as `sd.<area>.<Class>.<method>(...)`, called on an instance: `self`
//   and a `/` right after it dropped); the `params` names are the signatures'
//   parameters, each once, in first-appearance order or in an order that keeps
//   every signature's order; `summary` is one
//   sentence ending with a period; a class page's attributes and methods are
//   members of the class, a method's `signature` is `<method>(...)` in the
//   same form; `of` names a class page.
// - area overviews: unique group titles; `glance` slugs (warning if missing).
// - start pages: title, summary, order.
// Errors exit with status 1; warnings do not.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pnpm = path.join(SITE, 'node_modules', '.pnpm');
const pkg = (name) => {
  const dir = fs.readdirSync(pnpm).find((d) => d.startsWith(`${name.replace('/', '+')}@`));
  if (!dir) throw new Error(`${name} not found in node_modules/.pnpm (run pnpm install)`);
  return path.join(pnpm, dir, 'node_modules', name);
};
const YAML = (await import(pathToFileURL(path.join(pkg('yaml'), 'dist', 'index.js')))).default;

const CONTENT = path.join(SITE, 'src', 'content');
const inventory = JSON.parse(fs.readFileSync(path.join(SITE, 'notes', 'api-inventory.json'), 'utf8')).areas;
const invKey = (area) => (area === 'array' ? 'sd' : area);
const nsOf = (area) => (area === 'array' ? 'sd' : `sd.${area}`);

let errors = 0;
let warnings = 0;
const err = (file, msg) => {
  errors++;
  console.log(`ERR  ${path.relative(SITE, file)}: ${msg}`);
};
const warn = (file, msg) => {
  warnings++;
  console.log(`warn ${path.relative(SITE, file)}: ${msg}`);
};

function read(file) {
  const text = fs.readFileSync(file, 'utf8');
  const m = text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!m) {
    err(file, 'no frontmatter');
    return null;
  }
  try {
    return { data: YAML.parse(m[1]) ?? {}, body: text.slice(m[0].length) };
  } catch (e) {
    err(file, `YAML: ${e.message.split('\n')[0]}`);
    return null;
  }
}

/** "a, b=(1, 2), /, *, c" -> top-level pieces. */
function splitArgs(inner) {
  const out = [];
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
/** Parameter names of "(a, b=1, /, *, c)" in order, `*args`/`**kw` with their stars; `self` dropped. */
const paramNames = (sig) =>
  splitArgs(sig.slice(sig.indexOf('(') + 1, sig.lastIndexOf(')')))
    .filter((a) => a !== '/' && a !== '*')
    .map((a) => a.match(/^(\*{0,2}[A-Za-z_]\w*)/)?.[1] ?? a)
    .filter((n) => n !== 'self');
/**
 * The method as it is called on an instance: "(self, x, /)" -> "(x, /)",
 * "(self, /, *, axis=None)" -> "(*, axis=None)", "(self, /)" -> "()". A `/`
 * right after `self` marks only `self` as positional-only, so it goes too.
 * Annotations as the inventory prints them are kept.
 */
const dropSelf = (sig) => {
  const args = splitArgs(sig.slice(sig.indexOf('(') + 1, sig.lastIndexOf(')')));
  if (args[0] === 'self') {
    args.shift();
    if (args[0] === '/') args.shift();
  }
  return `(${args.join(', ')})`;
};

const KNOWN = {
  api: ['kind', 'title', 'names', 'group', 'order', 'signatures', 'summary', 'params', 'returns', 'raises', 'seeAlso', 'returnedBy', 'attributes', 'methods', 'of'],
  areas: ['title', 'ns', 'lead', 'glance', 'groups'],
  start: ['title', 'summary', 'order'],
};

function common(file, d, kind) {
  for (const k of Object.keys(d)) if (!KNOWN[kind].includes(k)) err(file, `unknown field "${k}"`);
  const text = JSON.stringify(d);
  if (/[가-힣]/.test(text)) err(file, 'Korean text in frontmatter');
}

const overviews = new Map();
function overview(area) {
  if (!overviews.has(area)) {
    const file = path.join(CONTENT, 'areas', `${area}.mdx`);
    overviews.set(area, fs.existsSync(file) ? read(file)?.data : undefined);
  }
  return overviews.get(area);
}

function checkApi(file, d) {
  const area = path.basename(path.dirname(file));
  const slug = path.basename(file, '.mdx');
  common(file, d, 'api');
  for (const k of ['kind', 'title', 'group', 'summary']) if (!d[k]) err(file, `missing "${k}"`);
  if (d.kind && !['function', 'class', 'method', 'concept'].includes(d.kind)) err(file, `kind "${d.kind}"`);
  const inv = inventory[invKey(area)]?.names;
  if (!inv) return err(file, `no area "${area}" in the inventory`);

  // group
  const ov = overview(area);
  if (!ov) err(file, `no overview src/content/areas/${area}.mdx`);
  else if (!(ov.groups ?? []).some((g) => g.title === d.group))
    err(file, `group "${d.group}" is not in areas/${area}.mdx (${(ov.groups ?? []).map((g) => g.title).join(', ')})`);

  // summary: one sentence with a final period
  if (d.summary) {
    if (!/\.$/.test(d.summary)) err(file, 'summary must end with a period');
    if (/\.\s+[A-Z]/.test(d.summary.replace(/`[^`]*`/g, '``').replace(/\b(e\.g|i\.e|etc|vs)\./g, ''))) warn(file, 'summary looks like more than one sentence');
  }

  // names
  const names = d.names ?? [];
  const member = (n) => {
    const [cls, m] = n.split('.');
    return inv[cls]?.members?.[m];
  };
  for (const n of names) if (!(n in inv) && !(n.includes('.') && member(n))) err(file, `"${n}" is not a public name of ${nsOf(area)}`);
  if (d.kind !== 'concept' && names.length === 0) err(file, 'names is empty');

  // signatures
  const sigs = d.signatures ?? [];
  if ((d.kind === 'function' || d.kind === 'method') && sigs.length === 0) err(file, 'a function or method page needs signatures');
  const prefix = `${nsOf(area)}.`;
  const sigParams = [];
  for (const s of sigs) {
    if (!s.startsWith(prefix) || !s.includes('(')) {
      err(file, `signature "${s}" does not start with ${prefix}`);
      continue;
    }
    const callee = s.slice(prefix.length, s.indexOf('('));
    const rest = s.slice(s.indexOf('('));
    let pinned;
    if (callee in inv) pinned = inv[callee].signature;
    else if (callee.includes('.') && member(callee)) pinned = member(callee).signature && dropSelf(member(callee).signature);
    else {
      err(file, `signature "${s}": ${callee} is not public`);
      continue;
    }
    if (pinned == null) err(file, `signature "${s}": the inventory has no signature for ${callee}`);
    else if (rest !== pinned) err(file, `signature "${s}" differs from the pinned ${prefix}${callee}${pinned}`);
    if (!names.includes(callee)) err(file, `signature of ${callee}, which is not in names`);
    for (const p of paramNames(rest)) if (!sigParams.includes(p)) sigParams.push(p);
  }

  // params are the signatures' parameters, each once: in the order they first
  // appear across the signatures, or in an order that keeps the order of every
  // signature (`correction` before `keepdims` for nanvar beside nansum)
  const params = (d.params ?? []).flatMap((p) => String(p.name ?? '').split(',').map((n) => n.trim()));
  for (const p of d.params ?? []) for (const k of ['name', 'type', 'desc']) if (!p[k]) err(file, `a param lacks "${k}"`);
  if (sigs.length > 0) {
    const same = params.length === sigParams.length && new Set(params).size === params.length && sigParams.every((p) => params.includes(p));
    const ordered = sigs.every((s) => {
      const own = paramNames(s.slice(s.indexOf('('))).filter((p) => params.includes(p));
      const pos = own.map((p) => params.indexOf(p));
      return pos.every((v, i) => i === 0 || pos[i - 1] < v);
    });
    if (!same || (params.join(',') !== sigParams.join(',') && !ordered))
      err(file, `params (${params.join(', ')}) do not match the signatures' parameters (${sigParams.join(', ')}, or an order that keeps each signature's)`);
  }
  for (const p of d.params ?? []) {
    if (p.kind && !['positional-only', 'keyword-only'].includes(p.kind)) err(file, `param ${p.name}: kind "${p.kind}"`);
    if (p.choices && !p.choices.every((c) => Array.isArray(c) && c.length === 2)) err(file, `param ${p.name}: choices must be [value, meaning] pairs`);
  }
  // kind tags and defaults agree with the first signature that has the parameter
  for (const s of sigs) {
    const inner = splitArgs(s.slice(s.indexOf('(') + 1, s.lastIndexOf(')')));
    const slash = inner.indexOf('/');
    const star = inner.findIndex((a) => a === '*' || /^\*[A-Za-z_]/.test(a));
    inner.forEach((a, i) => {
      const name = a.match(/^\*{0,2}([A-Za-z_]\w*)/)?.[1];
      const p = (d.params ?? []).find((q) => String(q.name).split(',').map((n) => n.trim().replace(/^\*+/, '')).includes(name));
      if (!name || !p || a.startsWith('*')) return;
      const want = i < slash ? 'positional-only' : star !== -1 && i > star ? 'keyword-only' : undefined;
      if ((p.kind ?? undefined) !== want) err(file, `param ${name}: kind should be ${want ?? 'absent'} (from ${s})`);
      const def = a.includes('=') ? a.slice(a.indexOf('=') + 1).trim() : undefined;
      if (def !== undefined && p.default !== undefined && String(p.default) !== def) err(file, `param ${name}: default "${p.default}", signature says ${def}`);
      if (def === undefined && p.default !== undefined) err(file, `param ${name} has a default but the signature has none`);
      if (def !== undefined && p.default === undefined) err(file, `param ${name}: default ${def} is missing`);
    });
  }

  // returns
  const rets = d.returns ?? [];
  if (rets.length > 1 && rets.some((r) => !r.name)) err(file, 'several returns need a name each');

  // classes
  if (d.kind === 'class') {
    const cls = inv[d.title];
    if (!cls || cls.kind !== 'class') err(file, `title "${d.title}" is not a public class`);
    else {
      for (const a of d.attributes ?? []) if (!cls.members?.[a.name]) err(file, `attribute ${a.name} is not a member of ${d.title}`);
      for (const m of d.methods ?? []) {
        const pinned = cls.members?.[m.name];
        if (pinned?.kind !== 'method') err(file, `method ${m.name} is not a method of ${d.title}`);
        else if (m.signature != null && pinned.signature && m.signature !== `${m.name}${dropSelf(pinned.signature)}`)
          err(file, `method ${m.name}: signature "${m.signature}", pinned ${m.name}${dropSelf(pinned.signature)}`);
      }
    }
  } else if ((d.attributes ?? []).length || (d.methods ?? []).length || (d.returnedBy ?? []).length) err(file, 'attributes, methods and returnedBy are for kind: class');
  if (d.kind === 'method') {
    if (!d.of) err(file, 'a method page needs "of"');
    else if (!fs.existsSync(path.join(path.dirname(file), `${d.of}.mdx`))) err(file, `of: no page ${area}/${d.of}`);
  }
  if (slug !== slug.trim() || /\s/.test(slug)) err(file, 'file name has spaces');
}

function checkArea(file, d) {
  common(file, d, 'areas');
  const area = path.basename(file, '.mdx');
  for (const k of ['title', 'ns', 'lead', 'groups']) if (!d[k]) err(file, `missing "${k}"`);
  if (d.ns && d.ns !== nsOf(area)) err(file, `ns "${d.ns}" should be ${nsOf(area)}`);
  const titles = (d.groups ?? []).map((g) => g.title);
  for (const g of d.groups ?? []) if (!g.title || !g.desc) err(file, 'every group needs title and desc');
  const dup = titles.filter((t, i) => titles.indexOf(t) !== i);
  if (dup.length) err(file, `duplicate groups: ${dup.join(', ')}`);
  for (const s of d.glance ?? []) {
    const target = s.includes('/') ? path.join(CONTENT, 'api', `${s}.mdx`) : path.join(CONTENT, 'api', area, `${s}.mdx`);
    if (!fs.existsSync(target)) warn(file, `glance "${s}" has no page yet`);
  }
}

function checkStart(file, d) {
  common(file, d, 'start');
  for (const k of ['title', 'summary']) if (!d[k]) err(file, `missing "${k}"`);
  if (typeof d.order !== 'number') err(file, 'order must be a number');
}

const all = () => {
  const list = [];
  for (const sub of ['api', 'areas', 'start']) {
    const walk = (dir) => {
      if (!fs.existsSync(dir)) return;
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) walk(p);
        else if (e.name.endsWith('.mdx')) list.push(p);
      }
    };
    walk(path.join(CONTENT, sub));
  }
  return list.sort();
};

const files = process.argv.length > 2 ? process.argv.slice(2).map((f) => path.resolve(f)) : all();
for (const file of files) {
  const page = read(file);
  if (!page) continue;
  const rel = path.relative(CONTENT, file).split(path.sep);
  if (rel[0] === 'api') checkApi(file, page.data);
  else if (rel[0] === 'areas') checkArea(file, page.data);
  else if (rel[0] === 'start') checkStart(file, page.data);
  else err(file, 'not under src/content/api, areas or start');
}
console.log(`${files.length} files, ${errors} errors, ${warnings} warnings`);
process.exit(errors ? 1 : 0);
