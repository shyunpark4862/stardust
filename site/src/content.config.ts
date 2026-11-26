import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

/**
 * An entry's id is its path without the extension ("linalg/solve"), so the
 * URL slug is always the file name. The default id would drop a file named
 * index.mdx down to its folder and lose the slug.
 */
const pathId = ({ entry }: { entry: string }) => entry.replace(/\.mdx$/, '');

/** Text fields in frontmatter take `code` and [links](/api/...) only (see src/data/inline.ts). */
const text = z.string().min(1);

const param = z
  .object({
    /** One name, or several that share a row ("x1, x2"); `*args` keeps its stars. */
    name: text,
    /** Type in words: "array of shape (..., n, n)", "float or None". */
    type: text,
    kind: z.enum(['positional-only', 'keyword-only']).optional(),
    /** The default as written in the signature; omitted when the parameter is required. */
    default: z.string().optional(),
    desc: text,
    /** Value | meaning rows, shown as a table under the description. */
    choices: z.array(z.tuple([z.string(), z.string()])).optional(),
  })
  .strict();

const ret = z.object({ name: z.string().optional(), type: text, desc: text }).strict();
const raise = z.object({ type: text, desc: text }).strict();
const attribute = z.object({ name: text, type: text, desc: text }).strict();
const method = z
  .object({
    name: text,
    /** Call form without the class: "summary()", "predict(x, /)". */
    signature: text,
    summary: text,
    /** Slug of the method's own page (kind: method), when it has one. */
    slug: z.string().optional(),
  })
  .strict();

/**
 * One file per reference page: src/content/api/<area>/<slug>.mdx.
 * The frontmatter is the reference card (site/WRITING.md §4); the body is
 * the description, examples and notes.
 */
const api = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/api', generateId: pathId }),
  schema: z
    .object({
      kind: z.enum(['function', 'class', 'method', 'concept']),
      /** The name, or a short English title for a family or concept page. */
      title: text,
      /** Every public name the page documents (class members as Class.member). */
      names: z.array(text).default([]),
      /** Must be one of the group titles in src/content/areas/<area>.mdx. */
      group: text,
      /** Position inside the group; ties fall back to the title. */
      order: z.number().default(100),
      /** Exactly the pinned signatures, with the full path ("sd.linalg.solve(a, b, /)"). */
      signatures: z.array(text).default([]),
      /** One imperative sentence (a noun phrase for classes), ending with a period. */
      summary: text,
      params: z.array(param).default([]),
      returns: z.array(ret).default([]),
      raises: z.array(raise).default([]),
      /** Same-area slugs or "area/slug". */
      seeAlso: z.array(text).default([]),
      /** Classes: slugs (same area or "area/slug") of the functions that return it. */
      returnedBy: z.array(text).default([]),
      attributes: z.array(attribute).default([]),
      methods: z.array(method).default([]),
      /** Method pages: slug of the class page (same area). */
      of: z.string().optional(),
    })
    .strict()
    .superRefine((d, ctx) => {
      if (d.kind === 'method' && !d.of) ctx.addIssue({ code: 'custom', message: 'a method page needs `of: <class slug>`' });
      if (d.kind !== 'method' && d.of) ctx.addIssue({ code: 'custom', message: '`of` is only for kind: method' });
      if (d.kind !== 'class' && (d.returnedBy.length || d.attributes.length || d.methods.length))
        ctx.addIssue({ code: 'custom', message: 'returnedBy, attributes and methods are only for kind: class' });
      if (d.returns.length > 1 && d.returns.some((r) => !r.name))
        ctx.addIssue({ code: 'custom', message: 'several returns (a tuple) need a name each' });
    }),
});

/** One overview per area: src/content/areas/<area>.mdx. */
const areas = defineCollection({
  loader: glob({ pattern: '*.mdx', base: './src/content/areas', generateId: pathId }),
  schema: z
    .object({
      /** "Linear algebra" */
      title: text,
      /** "sd.linalg" */
      ns: text,
      /** One to three sentences: what the area covers and what it does not. */
      lead: text,
      /** 4–8 slugs of the most-used pages, shown as cards. */
      glance: z.array(text).default([]),
      /** The area's groups, in the order the sidebar and the tables show them. */
      groups: z.array(z.object({ title: text, desc: text }).strict()).min(1),
    })
    .strict(),
});

/** Getting started pages: src/content/start/<slug>.mdx. */
const start = defineCollection({
  loader: glob({ pattern: '*.mdx', base: './src/content/start', generateId: pathId }),
  schema: z.object({ title: text, summary: text, order: z.number() }).strict(),
});

export const collections = { api, areas, start };
