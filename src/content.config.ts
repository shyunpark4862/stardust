import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const field = z.object({
  /** Type as written in the docs, e.g. "ndarray". */
  type: z.string(),
  desc: z.string(),
});

/**
 * One file per reference page: src/content/api/<area>/<slug>.mdx.
 *
 * A page documents one function, a family of functions that share one
 * explanation (sin, cos, tan ...), or a concept (broadcasting). The
 * frontmatter is the reference card; the body is the long-form text.
 */
const api = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/api' }),
  schema: z.object({
    kind: z.enum(['function', 'concept']).default('function'),
    /** Page heading: the function name, or a Korean title for families and concepts. */
    title: z.string(),
    /** Public names documented on this page. Empty for concepts. */
    names: z.array(z.string()).default([]),
    /** Heading the page is listed under on its area page. */
    group: z.string(),
    /** Position inside the group; ties fall back to the title. */
    order: z.number().default(100),
    /** One call signature per line. */
    signatures: z.array(z.string()).default([]),
    summary: z.string(),
    tag: z.string().optional(),
    params: z.array(field.extend({ name: z.string(), kind: z.string().optional() })).default([]),
    returns: field.optional(),
    raises: z.array(field).default([]),
    /** Slugs of related pages in the same area. */
    seeAlso: z.array(z.string()).default([]),
    /** Placeholder page: shows the "sample" notice. */
    sample: z.boolean().default(false),
  }),
});

export const collections = { api };
