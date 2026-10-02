// The terms of use: their text, their version, and remembering that the user accepted them.
// Raising TERMS_VERSION asks everyone once more on the next start.

import type { Lang } from '../../i18n'
import { isTauri, shell, type ShellState } from '../../shell/bridge'

export const TERMS_VERSION = 1
export const TERMS_DATE = '2026-10-02'

interface Section {
  title: string
  body: string[]
}

const ZH: Section[] = [
  {
    title: '一、条款的接受',
    body: [
      '本《使用条款》（以下简称“本条款”）是你与星籁（Stellaria，以下简称“本软件”）开发者之间就本软件的下载、安装和使用所订立的协议。',
      '你勾选同意、或以安装、启动、使用等方式实际使用本软件，即表示你已阅读、理解并同意受本条款约束。如你不同意本条款的任何内容，请立即停止使用并卸载本软件。',
    ],
  },
  {
    title: '二、软件的性质',
    body: [
      '本软件是一款在你的设备本地运行的歌声转换与音频创作工具，按“现状”和“现有”的基础提供。',
      '在适用法律允许的最大范围内，开发者不对本软件作任何明示或默示的保证，包括但不限于对适销性、特定用途适用性、不侵权、运行不中断或无错误、以及生成结果的准确性、质量或可用性的保证。',
      '本软件免费提供。任何人不得出售本软件或其副本、修改版本，也不得以本软件为主要内容收取费用，包括但不限于付费下载、付费整合包，以及代安装、代训练、代转换等收费服务。如果你是付费获得本软件的，请向卖家要求退款，并从官方地址免费下载。',
    ],
  },
  {
    title: '三、你的内容与责任',
    body: [
      '你通过本软件导入、处理、训练、生成、导出或传播的全部内容（包括但不限于声音素材、模型、音频和工程文件），均由你自行负责。',
      '你应确保对所使用的声音素材及其他内容享有合法权利，或已取得权利人（包括声音所属的自然人）的明确授权。',
      '你不得利用本软件从事任何违反法律法规或侵害他人合法权益的行为，包括但不限于：侵犯他人的声音、肖像、姓名、名誉、隐私等人格权益或著作权等知识产权；冒充他人；实施诈骗或其他违法犯罪活动；制作、传播违法或不良信息。',
      '你对外发布、传播使用本软件生成或合成的内容时，应当遵守适用的法律法规，并按规定以显著方式标明其为人工智能生成或合成的内容。',
    ],
  },
  {
    title: '四、第三方组件、模型与插件',
    body: [
      '本软件在运行过程中可能下载或调用第三方开源组件和预训练模型，它们分别受各自许可协议的约束，其中部分仅允许非商业用途。具体名称和许可见本软件“资源”页面。你应自行遵守相关许可。',
      '你自行安装或加载的第三方插件（包括 VST3、AU 等格式）、模型及其他内容，其来源、授权、合法性与安全性由你自行负责；本软件不对其进行审查，亦不作任何保证。',
    ],
  },
  {
    title: '五、数据与隐私',
    body: [
      '你的素材、模型和作品保存在你自己的设备上。本软件不收集、不上传你的个人数据或创作内容。',
      '本软件仅在下载运行环境、模型和组件，或执行你主动发起的联网操作时访问网络。',
    ],
  },
  {
    title: '六、责任限制',
    body: [
      '在适用法律允许的最大范围内，对于因使用或无法使用本软件而产生的任何直接、间接、附带、特殊、惩罚性或后果性损失（包括但不限于数据丢失、设备损坏、业务中断、利润损失，以及因你的内容或行为引起的第三方索赔），开发者均不承担责任。',
      '因你违反本条款或适用法律而导致开发者遭受任何第三方索赔、处罚或损失的，由你承担全部责任。',
    ],
  },
  {
    title: '七、条款的变更与效力',
    body: [
      '开发者可能根据需要更新本条款。条款更新后，本软件会在你下次启动时请你重新确认；你继续使用即视为接受更新后的条款。',
      '本条款任何条款被认定为无效或不可执行的，不影响其余条款的效力。',
    ],
  },
]

const EN: Section[] = [
  {
    title: '1. Acceptance',
    body: [
      'These Terms of Use (the “Terms”) are an agreement between you and the developers of Stellaria (the “Software”) covering the download, installation and use of the Software.',
      'By ticking “I agree”, or by installing, starting or otherwise using the Software, you confirm that you have read, understood and agree to be bound by these Terms. If you do not agree with any part of them, stop using the Software and uninstall it.',
    ],
  },
  {
    title: '2. The Software',
    body: [
      'The Software is a singing voice conversion and audio creation tool that runs locally on your device. It is provided “as is” and “as available”.',
      'To the maximum extent permitted by applicable law, the developers make no warranties of any kind, express or implied, including warranties of merchantability, fitness for a particular purpose, non-infringement, uninterrupted or error-free operation, or the accuracy, quality or usefulness of any output.',
      'The Software is free. No one may sell it, its copies or modified versions, or charge for anything that consists mainly of it, including paid downloads, paid bundles, and paid installation, training or conversion services. If you paid for the Software, ask the seller for a refund and download it free from the official address.',
    ],
  },
  {
    title: '3. Your content and responsibility',
    body: [
      'You are solely responsible for everything you import, process, train, generate, export or distribute with the Software, including voice material, models, audio and project files.',
      'You must hold the rights to the voice material and other content you use, or have the explicit permission of the rights holders, including the people whose voices they are.',
      'You must not use the Software in any way that breaks the law or infringes the rights of others, including their rights in their voice, likeness, name, reputation or privacy, or their copyright and other intellectual property; to impersonate anyone; to commit fraud or any other crime; or to create or spread unlawful content.',
      'When you publish or distribute content generated or synthesised with the Software, you must comply with applicable laws and regulations, including any requirement to label it clearly as AI-generated or synthesised.',
    ],
  },
  {
    title: '4. Third-party components, models and plug-ins',
    body: [
      'The Software may download or use third-party open-source components and pretrained models, each under its own licence; some allow non-commercial use only. Their names and licences are listed on the Software’s Resources page. You are responsible for complying with them.',
      'You are solely responsible for the origin, licensing, legality and safety of any third-party plug-ins (including VST3 and AU), models or other content you install or load. The Software does not review them and gives no warranty for them.',
    ],
  },
  {
    title: '5. Data and privacy',
    body: [
      'Your material, models and works are stored on your own device. The Software does not collect or upload your personal data or your creations.',
      'The Software accesses the network only to download its runtime, models and components, or for network actions you start yourself.',
    ],
  },
  {
    title: '6. Limitation of liability',
    body: [
      'To the maximum extent permitted by applicable law, the developers are not liable for any direct, indirect, incidental, special, punitive or consequential loss arising from the use of, or inability to use, the Software, including loss of data, damage to devices, business interruption, loss of profits, and third-party claims arising from your content or conduct.',
      'You are fully responsible for any third-party claim, penalty or loss the developers suffer because you breached these Terms or applicable law.',
    ],
  },
  {
    title: '7. Changes and severability',
    body: [
      'The developers may update these Terms. After an update, the Software asks you to confirm them again the next time it starts; continuing to use it means you accept the updated Terms.',
      'If any provision of these Terms is held invalid or unenforceable, the remaining provisions stay in effect.',
    ],
  },
]

export function termsSections(lang: Lang): Section[] {
  return lang === 'en' ? EN : ZH
}

const LOCAL_KEY = 'stellaria.termsAccepted'

/** The accepted version (the shell keeps it; the browser dev build uses localStorage). */
export function acceptedVersion(state: ShellState | null): number {
  if (isTauri) return state?.termsAccepted ?? 0
  try {
    return Number(localStorage.getItem(LOCAL_KEY) ?? 0) || 0
  } catch {
    return 0
  }
}

/** Remembers the acceptance of the current version; returns the new shell state (Tauri). */
export async function acceptTerms(): Promise<ShellState | null> {
  if (isTauri) return shell.acceptTerms(TERMS_VERSION)
  try {
    localStorage.setItem(LOCAL_KEY, String(TERMS_VERSION))
  } catch {
    /* private mode: asked again next time */
  }
  return null
}
