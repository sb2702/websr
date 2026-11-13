// Type declarations for Float16Array
// This enables TypeScript/Webpack to recognize Float16Array without errors

declare var Float16Array: {
  new(): Float16Array;
  new(length: number): Float16Array;
  new(array: ArrayLike<number>): Float16Array;
  new(buffer: ArrayBuffer, byteOffset?: number, length?: number): Float16Array;
  readonly prototype: Float16Array;
  readonly BYTES_PER_ELEMENT: number;
};

interface Float16Array {
  readonly length: number;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly buffer: ArrayBuffer;
  readonly BYTES_PER_ELEMENT: number;
  
  [index: number]: number;
  
  // Array-like methods
  copyWithin(target: number, start: number, end?: number): this;
  entries(): IterableIterator<[number, number]>;
  every(predicate: (value: number, index: number, array: Float16Array) => boolean, thisArg?: any): boolean;
  fill(value: number, start?: number, end?: number): this;
  filter(predicate: (value: number, index: number, array: Float16Array) => any, thisArg?: any): Float16Array;
  find(predicate: (value: number, index: number, obj: Float16Array) => boolean, thisArg?: any): number | undefined;
  findIndex(predicate: (value: number, index: number, obj: Float16Array) => boolean, thisArg?: any): number;
  forEach(callbackfn: (value: number, index: number, array: Float16Array) => void, thisArg?: any): void;
  includes(searchElement: number, fromIndex?: number): boolean;
  indexOf(searchElement: number, fromIndex?: number): number;
  join(separator?: string): string;
  keys(): IterableIterator<number>;
  lastIndexOf(searchElement: number, fromIndex?: number): number;
  map(callbackfn: (value: number, index: number, array: Float16Array) => number, thisArg?: any): Float16Array;
  reduce(callbackfn: (previousValue: number, currentValue: number, currentIndex: number, array: Float16Array) => number): number;
  reduce(callbackfn: (previousValue: number, currentValue: number, currentIndex: number, array: Float16Array) => number, initialValue: number): number;
  reduce<U>(callbackfn: (previousValue: U, currentValue: number, currentIndex: number, array: Float16Array) => U, initialValue: U): U;
  reduceRight(callbackfn: (previousValue: number, currentValue: number, currentIndex: number, array: Float16Array) => number): number;
  reduceRight(callbackfn: (previousValue: number, currentValue: number, currentIndex: number, array: Float16Array) => number, initialValue: number): number;
  reduceRight<U>(callbackfn: (previousValue: U, currentValue: number, currentIndex: number, array: Float16Array) => U, initialValue: U): U;
  reverse(): Float16Array;
  set(array: ArrayLike<number>, offset?: number): void;
  slice(start?: number, end?: number): Float16Array;
  some(predicate: (value: number, index: number, array: Float16Array) => boolean, thisArg?: any): boolean;
  sort(compareFn?: (a: number, b: number) => number): this;
  subarray(begin?: number, end?: number): Float16Array;
  toLocaleString(): string;
  toString(): string;
  valueOf(): Float16Array;
  values(): IterableIterator<number>;
  
  [Symbol.iterator](): IterableIterator<number>;
  readonly [Symbol.toStringTag]: "Float16Array";
}