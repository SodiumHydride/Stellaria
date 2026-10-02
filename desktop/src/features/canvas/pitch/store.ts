import { create } from 'zustand'

/** Which sing node's pitch editor is open (one at a time, over the creation workspace). */
export const usePitchEditor = create<{ nodeId: string | null; open: (nodeId: string) => void; close: () => void }>((set) => ({
  nodeId: null,
  open: (nodeId) => set({ nodeId }),
  close: () => set({ nodeId: null }),
}))
