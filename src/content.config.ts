import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const field = z.object({
  /** Type as written in the docs, e.g. "ndarray". */
  type: z.string(),
  desc: z.string(),
});

/**
 * One file per public function: src/content/api/<area>/<name>.mdx.
 * The frontmatter is the reference card; the body is the long-form text
 * (details, examples, notes).
 */
const api = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/api' }),
  schema: z.object({
    name: z.string(),
    /** Heading the function is listed under on its area page. */
    group: z.string(),
    /** Position inside the group; ties fall back to the name. */
    order: z.number().default(100),
    signature: z.string(),
    summary: z.string(),
    tag: z.string().optional(),
    params: z.array(field.extend({ name: z.string(), kind: z.string().optional() })).default([]),
    returns: field.optional(),
    raises: z.array(field).default([]),
    seeAlso: z.array(z.string()).default([]),
  }),
});

export const collections = { api };
