import { useCallback, useLayoutEffect, useRef } from 'react'

/** A ref that holds the value of the last committed render (for listeners and async callbacks). */
export function useLatest<T>(value: T) {
  const ref = useRef(value)
  // every render: the ref must follow the value, so there is nothing to compare
  useLayoutEffect(() => {
    ref.current = value
  })
  return ref
}

/**
 * A function with a stable identity that always runs the latest `fn`: for window listeners and
 * other handlers that read current state. Call it from events and effects, not during render.
 */
export function useEvent<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useLatest(fn)
  return useCallback((...args: A) => ref.current(...args), [ref])
}
