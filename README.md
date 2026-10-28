# 大肥鱼记忆大脑

本地 SQLite 长期记忆 + 向量召回 + 每轮重钉人格锚点，不依赖任何云服务

## 它解决什么问题

大多数记忆插件要么依赖云服务，要么把整个对话历史塞回 prompt。这个插件走另一条路：
- **本地 SQLite 五表**：事件 / 事实 / 画像 / 操作日志 / 状态
- **混合召回**：向量（余弦）+ 词法（中文 bigram 覆盖率），缺嵌入服务时自动降级为纯词法
- **对比式打分**：减去同批候选中位数，解决「无关内容余弦也有 0.5」导致的门限失效
- **会淡忘**：`importance × 0.99^天数`，`hits>0` 的豁免（被想起过的记得久）
- **人格锚点**：每轮把核心设定重钉在 system_prompt 末尾，抵抗长对话里的人设漂移
- **自动兜底捕捉**：模型忘了调用工具时，规则先接管（问句不会被误存）

## 安装

### 方式一：插件市场

在 AstrBot 面板 → 插件 → 插件市场 里搜索 `dafeiyu_brain`。

### 方式二：手动

```bash
cd AstrBot/data/plugins
git clone https://github.com/Caoran1314/astrbot_plugin_dafeiyu_brain
```

然后重启 AstrBot，或在面板里点「重载」。

## 配置

装好后在 面板 → 插件 → 大肥鱼记忆大脑 → 配置 里调。

| 配置项 | 默认 | 说明 |
|---|---|---|
| `threshold` | 0.14 | 召回门限 |
| `top_k` | 3 | 每次注入几条 |
| `embedding_enable` | false | 开启向量通道 |
| `embedding_api_base` | - | OpenAI 兼容的嵌入端点 |
| `embedding_model` | - | 如 `bge-m3` |
| `auto_capture` | true | 规则兜底自动记忆 |
| `owner_qq` | - | 主人 QQ 号，用于加权 |

## 命令

| 命令 | 作用 |
|---|---|
| `/brain` | 看统计 |
| `/brain search 关键词` | 手动查记忆 |
| `/brain export` | 导出 JSON（可迁移） |
| `/brain maintain` | 手动跑一次衰减清理 |

## 依赖

无（标准库 + 可选 numpy）

## 说明

- 只调用 AstrBot 自己的钩子，不修改框架代码
- 所有数据都在本地，不上传任何内容
- 适配平台：`aiocqhttp` / `webchat`

## License

MIT
