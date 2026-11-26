/// <reference types="astro/client" />

declare namespace App {
  interface Locals {
    /** The reference page being rendered (read by <Reference />). */
    apiPage?: {
      data: import('astro:content').CollectionEntry<'api'>['data'];
      /** Methods with a page of their own: name -> href. */
      methodHref: Record<string, string>;
    };
    /** The area overview being rendered (read by <FunctionTables /> and <ResultTable />). */
    apiArea?: string;
  }
}
