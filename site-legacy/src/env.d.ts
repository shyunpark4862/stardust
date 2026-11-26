/// <reference types="astro/client" />

declare namespace App {
  interface Locals {
    /** Frontmatter of the reference page being rendered (read by RefCard). */
    apiEntry?: import('astro:content').CollectionEntry<'api'>['data'];
  }
}
