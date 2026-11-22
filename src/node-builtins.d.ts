/**
 * Minimal ambient declarations for Node.js >= 23.6 / v26 built-in modules
 * used by `@opc/context-graph`, enabling zero-dependency `tsc --noEmit`
 * strict typechecking on a clean checkout without `node_modules`.
 */

interface Buffer extends Uint8Array {
  toString(encoding?: "utf8" | "hex" | "base64" | "base64url" | "ascii"): string;
  subarray(start?: number, end?: number): Buffer;
  indexOf(value: string | number | Uint8Array, byteOffset?: number): number;
}

declare var Buffer: {
  from(input: string, encoding?: "utf8" | "hex" | "base64" | "base64url" | "ascii"): Buffer;
  from(input: ArrayBuffer | Uint8Array | readonly number[]): Buffer;
  alloc(size: number, fill?: number): Buffer;
  concat(list: readonly Uint8Array[], totalLength?: number): Buffer;
  isBuffer(obj: unknown): obj is Buffer;
  byteLength(string: string | Uint8Array, encoding?: "utf8"): number;
};

declare var process: {
  argv: string[];
  env: Record<string, string | undefined>;
  exitCode?: number;
  stdin: import("node:stream").Readable;
  stdout: import("node:stream").Writable;
  stderr: import("node:stream").Writable;
  once(event: string, listener: (...args: unknown[]) => void): void;
  cwd(): string;
};

declare module "node:crypto" {
  export interface Hash {
    update(data: string | Uint8Array): this;
    digest(): Buffer;
    digest(encoding: "hex" | "base64" | "base64url"): string;
  }
  export function createHash(algorithm: string): Hash;
  export function createHmac(algorithm: string, key: string | Uint8Array): Hash;
  export function timingSafeEqual(a: Uint8Array, b: Uint8Array): boolean;
  export function randomUUID(): string;
}

declare module "node:fs" {
  export function mkdirSync(path: string, options?: { recursive?: boolean }): string | undefined;
  export function readFileSync(path: string, encoding: "utf8"): string;
  export function writeFileSync(path: string, data: string | Uint8Array): void;
  export function rmSync(path: string, options?: { recursive?: boolean; force?: boolean }): void;
  export function existsSync(path: string): boolean;
}

declare module "node:path" {
  export function dirname(p: string): string;
  export function resolve(...paths: string[]): string;
  export function join(...paths: string[]): string;
}

declare module "node:os" {
  export function tmpdir(): string;
}

declare module "node:url" {
  export function pathToFileURL(path: string): URL;
}

declare module "node:util" {
  export interface ParseArgsOptionConfig {
    type: "string" | "boolean";
    short?: string;
    multiple?: boolean;
  }
  export function parseArgs(config: {
    args?: string[];
    allowPositionals?: boolean;
    strict?: boolean;
    options?: Record<string, ParseArgsOptionConfig>;
  }): {
    values: Record<string, string | boolean | Array<string | boolean> | undefined>;
    positionals: string[];
  };
}

declare module "node:sqlite" {
  export type SQLInputValue = null | number | bigint | string | Uint8Array;
  export interface StatementResultingChanges {
    changes: number | bigint;
    lastInsertRowid: number | bigint;
  }
  export class StatementSync {
    run(...params: SQLInputValue[]): StatementResultingChanges;
    get(...params: SQLInputValue[]): unknown;
    all(...params: SQLInputValue[]): unknown[];
  }
  export class DatabaseSync {
    constructor(location: string);
    exec(sql: string): void;
    prepare(sql: string): StatementSync;
    close(): void;
  }
}

declare module "node:stream" {
  export class Readable {
    on(event: "data", listener: (chunk: Buffer | string) => void): this;
    on(event: "end" | "error" | string, listener: (...args: unknown[]) => void): this;
    once(event: string, listener: (...args: unknown[]) => void): this;
    off(event: "data", listener: (chunk: Buffer | string) => void): this;
    off(event: string, listener: (...args: unknown[]) => void): this;
    [Symbol.asyncIterator](): AsyncIterableIterator<Buffer | string>;
  }
  export class Writable {
    write(chunk: string | Uint8Array): boolean;
    end(chunk?: string | Uint8Array): this;
    on(event: string, listener: (...args: unknown[]) => void): this;
  }
  export class PassThrough extends Readable {
    write(chunk: string | Uint8Array): boolean;
    end(chunk?: string | Uint8Array): this;
  }
}

declare module "node:net" {
  export interface AddressInfo {
    address: string;
    family: string;
    port: number;
  }
}

declare module "node:http" {
  import type { Readable, Writable } from "node:stream";
  import type { AddressInfo } from "node:net";

  export interface IncomingHttpHeaders {
    origin?: string;
    accept?: string;
    "content-type"?: string;
    [header: string]: string | string[] | undefined;
  }

  export class IncomingMessage extends Readable {
    method?: string;
    url?: string;
    headers: IncomingHttpHeaders;
  }

  export class ServerResponse extends Writable {
    statusCode: number;
    writeHead(statusCode: number, headers?: Record<string, string | number>): this;
  }

  export class Server {
    listening: boolean;
    listen(port?: number, hostname?: string, listeningListener?: () => void): this;
    address(): AddressInfo | string | null;
    close(callback?: (err?: Error) => void): this;
    once(event: "error", listener: (err: Error) => void): this;
    off(event: "error", listener: (err: Error) => void): this;
  }

  export function createServer(
    requestListener: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>,
  ): Server;
}

declare module "node:test" {
  export interface TestContext {
    name: string;
  }
  export function describe(name: string, fn: () => void | Promise<void>): Promise<void>;
  export function it(name: string, fn: (t: TestContext) => void | Promise<void>): Promise<void>;
  export function test(name: string, fn: (t: TestContext) => void | Promise<void>): Promise<void>;
  export function before(fn: () => void | Promise<void>): void;
  export function after(fn: () => void | Promise<void>): void;
}

declare module "node:assert/strict" {
  export function equal<T>(actual: unknown, expected: T, message?: string): asserts actual is T;
  export function notEqual(actual: unknown, expected: unknown, message?: string): void;
  export function deepEqual<T>(actual: unknown, expected: T, message?: string): asserts actual is T;
  export function ok(value: unknown, message?: string): asserts value;
  export function match(string: string, regexp: RegExp, message?: string): void;
  export function throws(block: () => unknown, error?: RegExp | Function | object, message?: string): void;
  export function rejects(
    asyncFn: (() => Promise<unknown>) | Promise<unknown>,
    error?: RegExp | Function | object,
    message?: string,
  ): Promise<void>;
}
