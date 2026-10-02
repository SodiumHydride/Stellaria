// UI strings: [简体中文, English]. Keys are grouped by screen; feature files live in ./s/.

import { canvas } from './s/canvas'
import { create } from './s/create'
import { fx } from './s/fx'
import { material } from './s/material'
import { onboarding } from './s/onboarding'
import { pitch } from './s/pitch'
import { resources } from './s/resources'
import { settings } from './s/settings'
import { timeline } from './s/timeline'
import { voices } from './s/voices'
import { training } from './s/training'

const base = {
  // app
  'app.name': ['星籁', 'Stellaria'],

  // spaces
  'space.voices': ['角色', 'Voices'],
  'space.create': ['创作', 'Create'],
  'space.resources': ['资源', 'Resources'],
  'space.settings': ['设置', 'Settings'],

  // common
  'common.cancel': ['取消', 'Cancel'],
  'common.confirm': ['确定', 'OK'],
  'common.save': ['保存', 'Save'],
  'common.delete': ['删除', 'Delete'],
  'common.rename': ['重命名', 'Rename'],
  'common.close': ['关闭', 'Close'],
  'common.retry': ['重试', 'Retry'],
  'common.continue': ['继续', 'Continue'],
  'common.back': ['返回', 'Back'],
  'common.next': ['下一步', 'Next'],
  'common.done': ['完成', 'Done'],
  'common.loading': ['加载中…', 'Loading…'],
  'common.search': ['搜索', 'Search'],
  'common.all': ['全部', 'All'],
  'common.more': ['更多', 'More'],
  'common.copy': ['复制', 'Copy'],
  'common.copied': ['已复制', 'Copied'],
  'common.play': ['播放', 'Play'],
  'common.pause': ['暂停', 'Pause'],
  'common.stop': ['停止', 'Stop'],
  'common.advanced': ['高级设置', 'Advanced'],
  'common.reset': ['恢复默认', 'Reset'],
  'common.open_folder': ['打开所在文件夹', 'Show in folder'],
  'common.unknown': ['未知', 'Unknown'],
  'common.none': ['无', 'None'],
  'common.on': ['开', 'On'],
  'common.off': ['关', 'Off'],
  'common.minutes': ['{n} 分钟', '{n} min'],
  'common.items': ['{n} 项', '{n} items'],
  'common.error': ['出错了', 'Something went wrong'],
  'common.copy_error': ['复制错误信息', 'Copy error details'],
  'common.show_details': ['查看详情', 'Show details'],
  'common.hide_details': ['收起详情', 'Hide details'],
  'common.recommended': ['推荐', 'Recommended'],
  'common.optional': ['可选', 'Optional'],
  'common.required': ['必需', 'Required'],
  'common.select_files': ['选择文件', 'Choose files'],
  'common.select_folder': ['选择文件夹', 'Choose folder'],
  'common.undo': ['撤销', 'Undo'],
  'common.restore': ['恢复', 'Restore'],
  'common.new': ['新建', 'New'],
  'common.edit': ['编辑', 'Edit'],
  'common.export': ['导出', 'Export'],
  'common.import': ['导入', 'Import'],
  'common.exported': ['已导出', 'Exported'],
  'common.steps': ['步', 'steps'],
  'common.step_n': ['第 {n} 步', 'Step {n}'],

  // titlebar & palette
  'titlebar.search': ['搜索或执行命令…', 'Search or run a command…'],
  'titlebar.minimize': ['最小化', 'Minimize'],
  'titlebar.maximize': ['最大化', 'Maximize'],
  'titlebar.close': ['关闭', 'Close'],
  'palette.placeholder': ['输入角色、作品或操作…', 'Type a character, project or action…'],
  'palette.empty': ['没有找到匹配的内容', 'Nothing matches'],
  'palette.group.go': ['前往', 'Go to'],
  'palette.group.characters': ['角色', 'Characters'],
  'palette.group.projects': ['作品', 'Projects'],
  'palette.group.actions': ['操作', 'Actions'],
  'palette.new_character': ['新建角色', 'New character'],
  'palette.new_project': ['新建作品', 'New project'],
  'palette.import_material': ['给当前角色导入素材', 'Import material for this character'],
  'palette.pause_training': ['暂停训练「{name}」', 'Pause training “{name}”'],
  'palette.open_settings': ['打开设置', 'Open settings'],
  'palette.shortcuts': ['快捷键一览', 'Keyboard shortcuts'],

  // connection
  'conn.connecting': ['正在启动…', 'Starting…'],
  'conn.failed_title': ['连接不到星籁的后台服务', 'Can’t reach the Stellaria service'],
  'conn.failed_body': ['后台服务没有启动，或者启动时出了问题。可以重试；如果反复失败，复制日志发给开发者。', 'The background service didn’t start or crashed. Try again; if it keeps failing, copy the log and send it to the developer.'],
  'conn.restart': ['重启服务', 'Restart service'],
  'conn.copy_log': ['复制日志', 'Copy log'],
  'repair.title': ['你的 Python 环境变了', 'Your Python environment has changed'],
  'repair.checking': ['正在检查你的 Python…', 'Checking your Python…'],
  'repair.gone': ['找不到这个 Python 了，可能整个环境被删除或移动了。', 'This Python is gone; the environment may have been deleted or moved.'],
  'repair.version': ['这个 Python 变成了 {version}，星籁需要 3.10–3.12。', 'This Python is now {version}; Stellaria needs 3.10–3.12.'],
  'repair.torch': ['这个环境里的 PyTorch 不见了或装坏了，可能被升级或卸载过。', 'PyTorch in this environment is gone or broken; it may have been upgraded or removed.'],
  'repair.packages': ['这个环境更新过，星籁现在还缺 {n} 个包。可以重新补齐，照样装在星籁自己的目录里。', 'This environment was updated and Stellaria now lacks {n} packages. Add them again — into Stellaria’s own folder, as before.'],
  'repair.fine': ['你的 Python 看起来没问题，可能是别的原因。可以再检查补齐一遍后重试；还不行就复制日志发给开发者。', 'Your Python looks fine, so something else went wrong. Check and add the packages again, then retry; if it still fails, copy the log and send it to the developer.'],
  'repair.unknown': ['这个 Python 检查不了。', 'This Python couldn’t be checked.'],
  'repair.never_changed': ['星籁不会去改你的环境，下面任选一种方式继续。', 'Stellaria never changes your environment; pick a way to continue below.'],
  'repair.refill': ['重新补齐', 'Add them again'],
  'repair.rebuild': ['再检查补齐一遍', 'Check and add again'],
  'repair.use_managed': ['改用星籁自带的环境', 'Use Stellaria’s own runtime'],
  'repair.other_python': ['换一个 Python', 'Choose another Python'],
  'repair.choose_runtime': ['重新选择运行环境', 'Choose a runtime again'],
  'conn.offline': ['与后台的连接断开了，正在重连…', 'Connection lost, reconnecting…'],

  // dock
  'dock.idle': ['一切就绪', 'All quiet'],
  'dock.tasks': ['{n} 个任务进行中', '{n} tasks running'],
  'dock.queued': ['{n} 个排队', '{n} queued'],
  'dock.title': ['任务', 'Tasks'],
  'dock.empty': ['现在没有任务。整理素材、训练、渲染都会出现在这里。', 'No tasks. Organizing, training and rendering will show up here.'],
  'dock.clear': ['清除已完成', 'Clear finished'],
  'dock.cancel': ['取消任务', 'Cancel task'],
  'dock.view_log': ['查看日志', 'View log'],
  'dock.eta': ['约 {time}', '~{time} left'],
  'dock.waiting_gpu': ['等待显卡空闲', 'Waiting for the GPU'],
  'dock.gpu': ['显卡', 'GPU'],

  // job status
  'job.status.queued': ['排队中', 'Queued'],
  'job.status.waiting_gpu': ['等待显卡', 'Waiting for GPU'],
  'job.status.running': ['进行中', 'Running'],
  'job.status.stopping': ['正在停止', 'Stopping'],
  'job.status.succeeded': ['已完成', 'Done'],
  'job.status.failed': ['失败', 'Failed'],
  'job.status.cancelled': ['已取消', 'Cancelled'],

  // toasts
  'toast.job_done': ['{title} 完成', '{title} finished'],
  'toast.job_failed': ['{title} 失败', '{title} failed'],
  'toast.saved': ['已保存', 'Saved'],

  // drop overlay
  'crash.title': ['这里出了点问题', 'Something went wrong here'],
  'crash.body': ['你的作品和训练不受影响，都已经保存。重新加载通常就能恢复。', 'Your projects and training are safe and saved. Reloading usually fixes it.'],
  'crash.reload': ['重新加载', 'Reload'],
  'crash.home': ['回到首页', 'Go home'],
  'crash.copy': ['复制错误信息', 'Copy details'],
  'crash.copied': ['已复制', 'Copied'],
  'drop.title': ['松手即可导入', 'Drop to import'],
  'drop.to_project_audio': ['作为伴奏 / 音频（不分离）', 'As a backing track / audio (no split)'],
  'drop.to_character': ['加入「{name}」的素材', 'Add to “{name}” material'],
  'drop.new_character': ['新建角色并导入', 'Create a character from these'],
  'drop.to_project': ['作为歌曲导入（可分离人声）', 'As a song (can be split)'],
  'drop.new_project': ['新建作品并导入', 'New project from these'],
  'drop.hint': ['支持 wav · flac · mp3 · m4a · 视频，以及整个文件夹', 'wav · flac · mp3 · m4a · video files and whole folders'],
  // shortcuts
  'shortcuts.title': ['快捷键', 'Keyboard shortcuts'],
  'shortcuts.palette': ['打开命令面板', 'Open command palette'],
  'shortcuts.spaces': ['切换 角色 / 创作 / 资源', 'Switch Voices / Create / Resources'],
  'shortcuts.play': ['播放 / 暂停所选', 'Play / pause selection'],
  'shortcuts.review': ['素材审听：上一条 / 下一条', 'Review: previous / next clip'],
  'shortcuts.trash': ['素材审听：移到回收站', 'Review: move to trash'],
  'shortcuts.val': ['素材审听：设为验证集', 'Review: toggle validation'],
  'shortcuts.split': ['素材审听：在播放位置切开', 'Review: split at playhead'],
  'shortcuts.add': ['添加节点', 'Add a node'],
  'shortcuts.render': ['运行所选', 'Run selection'],
  'shortcuts.run_all': ['全部运行', 'Run everything'],
  'shortcuts.undo': ['撤销', 'Undo'],
  'shortcuts.redo': ['重做', 'Redo'],
  'shortcuts.copy_paste': ['复制 / 粘贴', 'Copy / paste'],
  'shortcuts.variant': ['再来一版 / 创建副本', 'New variant / duplicate'],
  'shortcuts.mix': ['把所选做成混音', 'Mix the selection'],
  'shortcuts.bypass': ['绕过效果', 'Bypass effects'],
  'shortcuts.frame': ['框成一组', 'Frame selection'],
  'mm.text': ['还缺 {n} 个必需的模型，训练和演唱前需要准备好', '{n} required models are still missing; training and singing need them'],
  'mm.go': ['去下载或导入', 'Download or import'],
  'shortcuts.rename': ['重命名节点', 'Rename node'],
  'shortcuts.port_play': ['试听端口（悬停端口也能看到波形）', 'Play a port (hover a port to see its waveform)'],
  'shortcuts.click': ['点击端口', 'Click a port'],
  'shortcuts.drag': ['拖动', 'Drag'],
  'shortcuts.compare_n': ['对比时切到第 N 个', 'Switch to take N while comparing'],
  'shortcuts.arrange': ['整理布局', 'Tidy up'],
  'shortcuts.compare': ['A/B 对比切换', 'Flip A/B'],
  'shortcuts.fit': ['适应窗口', 'Fit view'],
  'shortcuts.inspector': ['显示 / 隐藏右侧面板', 'Show / hide the side panel'],
  'shortcuts.delete': ['删除所选', 'Delete selection'],
  // close
  'close.title': ['还有任务在进行', 'Work is still in progress'],
  'close.body': ['关闭窗口后，训练和其他任务会在后台继续，星籁会待在托盘里。也可以先暂停训练再退出，进度会被保存。', 'Closing the window keeps training and other tasks running in the background from the tray. Or pause training and quit — progress is saved.'],
  'close.hide': ['在后台继续', 'Keep running in background'],
  'close.quit': ['暂停并退出', 'Pause and quit'],
  'close.setup_title': ['还在安装', 'Setup is still installing'],
  'close.setup_body': ['现在退出会停下正在进行的安装和下载。已经下载的部分会留着，下次打开星籁可以接着来。', 'Quitting now stops what is installing or downloading. What has been downloaded is kept, so the next launch can pick it up.'],
  'close.setup_quit': ['退出', 'Quit'],
  'close.setup_stay': ['继续安装', 'Keep installing'],
} satisfies Record<string, [string, string]>

export const strings = { ...base, ...voices, ...material, ...training, ...create, ...canvas, ...resources, ...settings, ...onboarding, ...fx, ...timeline, ...pitch }

export type StringKey = keyof typeof strings
