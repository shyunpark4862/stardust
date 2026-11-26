/**
 * The parts of the reference block a page has, in the order <Reference />
 * draws them. The page route builds its table of contents from the same list.
 */
interface RefData {
  params: unknown[];
  returns: unknown[];
  raises: unknown[];
  attributes: unknown[];
  methods: unknown[];
}

export function referenceSections(d: RefData): { slug: string; text: string }[] {
  return [
    d.params.length > 0 && { slug: 'parameters', text: 'Parameters' },
    d.returns.length > 0 && { slug: 'returns', text: 'Returns' },
    d.raises.length > 0 && { slug: 'raises', text: 'Raises' },
    d.attributes.length > 0 && { slug: 'attributes', text: 'Attributes' },
    d.methods.length > 0 && { slug: 'method-list', text: 'Methods' },
  ].filter((s): s is { slug: string; text: string } => Boolean(s));
}
