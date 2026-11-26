export interface NavItem {
  label: string;
  href: string;
  badge?: string;
  /** Show the label in the code font (function names). */
  mono?: boolean;
  /** Public names on a family page: shown under the label and matched by the filter. */
  names?: string[];
  /** A link to another site. */
  external?: boolean;
  /** A shorter label for narrow screens (the top navigation). */
  short?: string;
}
export interface NavGroup {
  title: string;
  items: NavItem[];
}

/** One step of the path shown above a page title. */
export interface Crumb {
  label: string;
  href?: string;
}

/** The area a reference page belongs to, and where the other areas are. */
export interface AreaNav {
  current: { key: string; name: string; ns: string; href: string };
  all: { key: string; name: string; ns: string; href?: string }[];
}

export const GITHUB = 'https://github.com/shyunpark4862/stardust';

/** Top navigation. One entry per site section, then the repository. */
export const topNav: NavItem[] = [
  { label: 'Get started', href: '/start/', short: 'Start' },
  { label: 'API', href: '/api/' },
  { label: 'Index', href: '/api/all/' },
  { label: 'GitHub', href: GITHUB, external: true },
];

/**
 * The top-navigation entry a page belongs to: the longest in-site href the
 * path starts with (so /api/all/ is "Index", not "API").
 */
export function currentTop(path: string): string | undefined {
  return topNav
    .filter((n) => !n.external && path.startsWith(n.href))
    .sort((a, b) => b.href.length - a.href.length)[0]?.href;
}

/** Sidebar trees, keyed by site section (pages may pass their own). */
export const sidebars: Record<string, NavGroup[]> = {};

export const areas = [
  { key: 'array', name: 'Arrays', ns: 'sd', desc: 'N-dimensional arrays, element-wise math, reductions, indexing, sorting', status: 'done' },
  { key: 'linalg', name: 'Linear algebra', ns: 'sd.linalg', desc: 'Decompositions, linear systems, norms, structured matrices, matrix functions', status: 'done' },
  { key: 'sparse', name: 'Sparse matrices', ns: 'sd.sparse', desc: 'COO, CSR, CSC and DOK formats, sparse products, direct and iterative solvers, eigenvalues', status: 'done' },
  { key: 'frame', name: 'Data frames', ns: 'sd.frame', desc: 'Series and DataFrame, missing values, group-by, joins, CSV input', status: 'done' },
  { key: 'special', name: 'Special functions', ns: 'sd.special', desc: 'Error functions, gamma and beta, incomplete integrals, Bessel and hypergeometric functions', status: 'done' },
  { key: 'stats', name: 'Statistics', ns: 'sd.stats', desc: 'Univariate, multivariate and matrix distributions, descriptive statistics', status: 'done' },
  { key: 'htest', name: 'Hypothesis tests', ns: 'sd.htest', desc: 't-tests and ANOVA, goodness of fit, rank tests, correlation, contingency tables, multiple comparisons, resampling', status: 'done' },
  { key: 'regression', name: 'Regression', ns: 'sd.regression', desc: 'Linear models, GLMs, regularized, robust and quantile regression, nonlinear least squares', status: 'done' },
  { key: 'optimize', name: 'Optimization', ns: 'sd.optimize', desc: 'Scalar roots and minima, multivariate minimization, least squares, systems of equations', status: 'done' },
  { key: 'integrate', name: 'Integration', ns: 'sd.integrate', desc: 'Adaptive quadrature, nested and multidimensional integrals, fixed and sampled rules', status: 'done' },
  { key: 'differentiate', name: 'Differentiation', ns: 'sd.differentiate', desc: 'Numerical derivatives, Jacobians and Hessians', status: 'done' },
  { key: 'random', name: 'Random numbers', ns: 'sd.random', desc: 'Counter-based generators, reproducible streams, sampling from distributions', status: 'done' },
  { key: 'spatial', name: 'Spatial', ns: 'sd.spatial', desc: 'Distance matrices, nearest neighbors, k-d trees, Delaunay triangulation, convex hulls, Voronoi diagrams', status: 'done' },
  { key: 'fft', name: 'Fourier transforms', ns: 'sd.fft', desc: 'Complex, real and Hermitian FFTs, DCT and DST, multidimensional transforms, FFTLog', status: 'done' },
  { key: 'interpolate', name: 'Interpolation', ns: 'sd.interpolate', desc: 'Piecewise polynomials and splines, barycentric and rational interpolation, gridded and scattered data', status: 'done' },
  { key: 'smooth', name: 'Smoothing', ns: 'sd.smooth', desc: 'LOWESS and LOESS, smoothing splines, kernel regression, kernel density estimation, Savitzky–Golay filters', status: 'done' },
  { key: 'graph', name: 'Graphs', ns: 'sd.graph', desc: 'Traversal and components, shortest paths, spanning trees, flows and matchings, centrality', status: 'done' },
  { key: 'plot', name: 'Plotting', ns: 'sd.plot', desc: 'Function plots, statistical graphics, 3D surfaces, PDF, SVG and PNG output', status: 'done' },
] as const;
