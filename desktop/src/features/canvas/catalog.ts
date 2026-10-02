// What can be put on the canvas: one entry per kind of node, for the add menu, node headers
// and context menus.

import {
  Blend,
  Combine,
  Disc3,
  DropletOff,
  FileAudio,
  FileDown,
  Frame,
  Gauge,
  Layers,
  ListMusic,
  Music,
  Scissors,
  SlidersHorizontal,
  Split,
  StickyNote,
  type LucideIcon,
} from 'lucide-react'
import type { CanvasNode, ProcessOp } from '../../api/types'
import { isMix, isProcess } from './model'

export type CatalogId = 'song' | 'audio' | 'mix' | 'bus' | ProcessOp | 'note' | 'group'
export type CatalogGroup = 'sources' | 'edit' | 'sound' | 'mixing' | 'output' | 'organize'

export interface CatalogEntry {
  id: CatalogId
  group: CatalogGroup
  icon: LucideIcon
  /** takes audio in (can be wired from a port / inserted on an edge) */
  input: 0 | 1 | 'many'
  /** search words besides the label (pinyin initials, English) */
  keywords: string
}

export const CATALOG: CatalogEntry[] = [
  { id: 'song', group: 'sources', icon: Music, input: 0, keywords: 'song import gequ daoru 歌曲 导入' },
  { id: 'audio', group: 'sources', icon: FileAudio, input: 0, keywords: 'audio backing instrumental banzou yinpin 伴奏 音频 和声' },
  { id: 'trim', group: 'edit', icon: Scissors, input: 1, keywords: 'trim cut region jianqie 剪切 截取 片段' },
  { id: 'split', group: 'edit', icon: Split, input: 1, keywords: 'split sections parts fenduan 分段 段落' },
  { id: 'separate', group: 'edit', icon: Layers, input: 1, keywords: 'separate stems demucs fenli 分离 分轨 鼓 贝斯' },
  { id: 'dereverb', group: 'edit', icon: DropletOff, input: 1, keywords: 'dereverb echo dry reverb quhunxiang huisheng gansheng 去混响 回声 混响 干声' },
  { id: 'comp', group: 'edit', icon: Combine, input: 'many', keywords: 'comp duet choose best pinjie 拼接 择优 对唱' },
  { id: 'concat', group: 'edit', icon: ListMusic, input: 'many', keywords: 'concat medley chuanlian 串联 串烧' },
  { id: 'pitch', group: 'sound', icon: Gauge, input: 1, keywords: 'pitch tempo key shift biandiao biansu 变调 变速 升降调' },
  { id: 'fx', group: 'sound', icon: SlidersHorizontal, input: 1, keywords: 'fx effects vst eq reverb xiaoguo 效果 混响 均衡 插件' },
  { id: 'mix', group: 'mixing', icon: Disc3, input: 'many', keywords: 'mix auto hunyin 混音 一键' },
  { id: 'bus', group: 'mixing', icon: Blend, input: 'many', keywords: 'bus premix sum yuhun 预混 总线 叠加' },
  { id: 'export', group: 'output', icon: FileDown, input: 1, keywords: 'export save daochu 导出 保存' },
  { id: 'note', group: 'organize', icon: StickyNote, input: 0, keywords: 'note bianqian 便签 备注' },
  { id: 'group', group: 'organize', icon: Frame, input: 0, keywords: 'group frame fenzu 分组 框' },
]

export const GROUP_ORDER: CatalogGroup[] = ['sources', 'edit', 'sound', 'mixing', 'output', 'organize']

export const entryOf = (id: CatalogId) => CATALOG.find((e) => e.id === id)!

/** The catalog entry a node was made from (for its icon and name). */
export function catalogIdOf(n: CanvasNode): CatalogId | null {
  if (n.type === 'source') return 'song'
  if (n.type === 'audio') return 'audio'
  if (isMix(n)) return n.data.processor === 'bus' ? 'bus' : 'mix'
  if (isProcess(n)) return n.data.op
  if (n.type === 'note' || n.type === 'group') return n.type
  return null
}
