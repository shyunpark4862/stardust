/**
 * The site can be served under a sub-path (GitHub Pages serves it at
 * /stardust/), set by `base` in astro.config.mjs. Paths in the source are
 * written from the site root ("/api/linalg/"); these helpers put the base in
 * front where a link is written out, and take it off where the page's own
 * path is compared with such a path.
 */

/** The base without its trailing slash: "" at the root, "/stardust" under a sub-path. */
const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');

/** A root-relative path ("/api/") with the base in front; anything else (#anchor, other hosts) unchanged. */
export function url(path: string): string;
export function url(path: string | undefined): string | undefined;
export function url(path: string | undefined): string | undefined {
  return path && path.startsWith('/') && !path.startsWith('//') ? BASE + path : path;
}

/** A page's own path (Astro.url.pathname) without the base, to compare with paths written from the root. */
export function unbase(pathname: string): string {
  return BASE && (pathname === BASE || pathname.startsWith(`${BASE}/`)) ? pathname.slice(BASE.length) || '/' : pathname;
}
