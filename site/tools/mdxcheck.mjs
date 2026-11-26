// MDX syntax check for the site's pages, without a full build.
//
//   node tools/mdxcheck.mjs [files ...]      (from site/; no files: every page)
//
// Compiles each body with the same Markdown extensions the site uses (GFM,
// math) and checks the placement components: an API page whose frontmatter
// has a reference block must place <Reference /> once; an area overview must
// place <FunctionTables /> once; every component used must be one the page
// imports or the template provides.
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
const { compile } = await import(pathToFileURL(path.join(pkg('@mdx-js/mdx'), 'index.js')));
const remarkGfm = (await import(pathToFileURL(path.join(SITE, 'node_modules', 'remark-gfm', 'index.js')))).default;
const remarkMath = (await import(pathToFileURL(path.join(SITE, 'node_modules', 'remark-math', 'index.js')))).default;
const YAML = (await import(pathToFileURL(path.join(pkg('yaml'), 'dist', 'index.js')))).default;

const CONTENT = path.join(SITE, 'src', 'content');
// Components each template passes to the page body.
const PROVIDED = { api: ['Reference'], areas: ['FunctionTables', 'ResultTable'], start: ['Callout'] };

const all = () => {
  const list = [];
  const walk = (dir) => {
    if (!fs.existsSync(dir)) return;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith('.mdx')) list.push(p);
    }
  };
  for (const sub of ['api', 'areas', 'start']) walk(path.join(CONTENT, sub));
  return list.sort();
};

/** Uses of <Name ...> outside fenced code. */
function components(body) {
  const used = [];
  let fence = false;
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (fence) continue;
    for (const m of line.matchAll(/<([A-Z]\w*)\b/g)) used.push(m[1]);
  }
  return used;
}

let bad = 0;
const files = process.argv.length > 2 ? process.argv.slice(2).map((f) => path.resolve(f)) : all();
for (const file of files) {
  const rel = path.relative(SITE, file);
  const text = fs.readFileSync(file, 'utf8');
  const m = text.match(/^---\n([\s\S]*?)\n---\n?/);
  const body = m ? text.slice(m[0].length) : text;
  const section = path.relative(CONTENT, file).split(path.sep)[0];
  const problems = [];
  try {
    await compile(body, { remarkPlugins: [remarkGfm, remarkMath] });
  } catch (e) {
    problems.push(e.message.split('\n')[0]);
  }
  const used = components(body);
  const imported = [...body.matchAll(/^import\s+(\w+)\s+from/gm)].map((x) => x[1]);
  for (const c of new Set(used)) if (!imported.includes(c) && !(PROVIDED[section] ?? []).includes(c)) problems.push(`<${c} /> is neither imported nor provided`);
  const count = (c) => used.filter((u) => u === c).length;
  if (section === 'api') {
    let d = {};
    try {
      d = (m && YAML.parse(m[1])) ?? {};
    } catch {}
    const card = ['params', 'returns', 'raises', 'attributes', 'methods'].some((k) => (d[k] ?? []).length > 0);
    if (card && count('Reference') !== 1) problems.push(`<Reference /> placed ${count('Reference')} times (needs 1)`);
    if (!card && count('Reference') > 0) problems.push('<Reference /> without params, returns, raises, attributes or methods');
  }
  if (section === 'areas' && count('FunctionTables') !== 1) problems.push(`<FunctionTables /> placed ${count('FunctionTables')} times (needs 1)`);
  if (/^#\s/m.test(body.replace(/```[\s\S]*?```/g, ''))) problems.push('a level-1 heading in the body (the template writes the title)');
  if (problems.length) {
    bad++;
    for (const p of problems) console.log(`ERR  ${rel}: ${p}`);
  } else console.log(`ok   ${rel}`);
}
console.log(`${files.length} files, ${bad} with problems`);
process.exit(bad ? 1 : 0);
