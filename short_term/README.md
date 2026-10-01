# short_term —— 档 A + 档 D

短期目标：**一个能出图、出对图、并且能局部控制的编辑器。**
两条路线，零训练先行：

| 档 | 是什么 | 成本 | 在哪 |
|---|---|---|---|
| **A** | 把已经在跑的噪声侧编辑器包成 CLI | 小时 | `edit.py --guide none` |
| **D** | 推理期引导（PiGDM 式），mask 当逐坐标权重 | 零训练 | `edit.py --guide target/source/both` |

---

## 0. 用任意一张图片，现在能做什么？（先回答这个）

**能，但有一个硬前提：你必须能给出"这块区域该长成什么样"的参考内容。**
只有"图 + mask + 一句指令"**还不行** —— 那正是缺 Y 的情况。

证据就是这两行（同一张真照片、同一句指令、同一个 mask、同一 seed）：

```
                                d_in   d_out_c   sat_in  exp_gain
有参考（--guide both）          0.075   0.0166    1.053    +0.788   ← 追平真值
无参考（--guide none，只有指令） 0.440   0.4474    0.281    -5.413   ← 背景被毁、区域漂白
(数据集真值)                    0.070   0.0069    1.126     1.000
```

**输入清单**（以 `--guide both` 为例）：

| 参数 | 是什么 | 从哪来 |
|---|---|---|
| `--image` | 源图 | 任意图片；会被 resize 到 `--size`（默认 512，**方形**） |
| `--mask` | 白 = 可编辑 | 手画 / SAM（`region/segment.py`）/ 数据集 |
| `--guide-image` | **参考图：mask 区域该长成什么样** | 见下表 —— **这是关键缺失项** |
| `--instruction` | 指令 | 仍然要：它驱动基底场，应与参考内容一致 |
| 超参 | `--beta 5 --kappa 0.6 --kappa-out 1.0 --kappa-mode flat` | **这一组已经实测过，别改** |
| `--steps 50 --image-guidance 1.5 --guidance 7.5 --seed 0` | 仓库评测口径 | 默认值 |

**参考图从哪来**（四种，后两种**没测过**）：

| 来源 | 可行性 |
|---|---|
| ① 从别的照片取一块同类的区域 | ✅ 最直接，就是"把 A 改成 B 那样" |
| ② 同一张图里别处的干净区域 | ✅ 补洞/替换，不需要外部素材 |
| ③ **生成**：T2I 生成目标内容 → 裁进 mask → 当 Y | ⚠️ **已实现（`propose_y.py`），但有效性未测** —— 见 §0.1 |
| ④ 用别的编辑器产候选 | ❌ **不推荐**：实测 tier A 的候选 `exp_gain = −5.4`，而 Gate C 上那三方编辑器 oracle mask 下也只有 22% clean |

### 0.1 两条产品线

| 线 | 用户提供 | 谁产 Y | 状态 |
|---|---|---|---|
| **1** | 图 + mask + **参考图** + 指令 | 用户 | ✅ **已测**：`d_in` 1.07× 真值、`sat_in` 0.94× 真值、`exp_gain` +0.788 |
| **2** | 图 + mask + **文字指令** | `propose_y.py`（T2I） | ⚠️ **管道已建，有效性未测** |

**两条线共用同一条命令**——区别只在 `--guide-image` 是谁给的：

```bash
# 线 1：用户给参考图
python short_term/edit.py --image src.png --mask m.png \
    --guide-image REF.png --instruction "<instruction>" \
    --guide both --kappa 0.6 --kappa-out 1.0 --kappa-mode flat --out out.png

# 线 2：先生成 Y，再跑*同一条*命令
python short_term/propose_y.py --image src.png --mask m.png \
    --instruction "<instruction>" \
    --y-prompt "a colourful hot air balloon floating in a blue sky" \
    --out y.png
python short_term/edit.py --image src.png --mask m.png \
    --guide-image y.png --instruction "<instruction>" \
    --guide both --kappa 0.6 --kappa-out 1.0 --kappa-mode flat --out out.png
```

`propose_y.py` 的关键设计与边界：

- **Y 在 mask 外恒等于源图**（逐像素，精确）。这是硬不变量：否则里面那一拉和外面那一拉
  会互相打架。自检对**四种策略**都断言了它，我也在写出的文件上独立复核过
  （`max|Y−src| = 0.000000`）。
- **对齐是这个脚本解决不了的事**。512 生成出来的热气球并不在你这张照片需要它的位置。
  `--strategy` 给了四种放置方式（`bbox` / `center` / `region` / `full`），
  **四种都是猜的**。
- **`--y-prompt` 应该给，不要用指令当 T2I 提示词**。`"A hot air balloon should also be
  floating."` 描述的是一次**改动**，不是一张**图**。脚本在缺 `--y-prompt` 时会打印这个警告。
- **`--variant` 就是"Y 要多好"的测量工具**（`solid` / `blur:S` / `noise:S` /
  `shift:PX` / `scale:F`）。`shift` 与 `scale` 模拟生成 Y 真实会有的对齐误差，
  `blur`/`noise` 模拟质量损失，`solid` 是"Y 什么都没说"的对照（应当退化成无引导的结果）。
  **先用它决定线 2 值不值得做，再去调 T2I 提示词。**

**线 2 的验收标准**（写下来再看）：
`d_in/tgt` ≈ 1.0、`d_out_c` < 0.02、`sat_in/tgt` ≥ 0.8、`exp_gain` > 0.5。
达不到就说明"生成一个粗略对位的 Y"不够用，**线 2 需要训练的 Y-proposer（档 B）**，
而不是继续调提示词。

### 0.2 交互模型：**在 patch 上圈出物体，然后把它拖到位置上**

#### 0.2.0 画框形状（默认 4:3，不再是方的）

原图窗口以前**永远是正方形**：`load_image` 把任何图 resize 成 `size×size`，网页上
`cropBox` 也无条件裁成方的。于是一张 4:3 的照片**左右两边被裁掉**（或勾了 squash 就被压扁），
而这完全没有必要。

现在画框是 **W×H**，网页左上角有选择器：

| 选项 | 像素 | 比例 |
|---|---|---|
| **4:3（默认）** | **512×384** | 1.33 |
| 1:1 | 512×512 | 1.00 |
| 3:4 | 384×512 | 0.75 |
| 16:9 | 512×288 | 1.78 |

- **上传的图按画框比例裁**（不再是裁成正方形）：4:3 的照片放进 4:3 画框**整张都用上**，
  不裁也不压。这不是显示器上的拉伸——mask、Y、模型输入**全都是同一个 W×H**。
- 命令行同理：`--size 512x384`（**宽在前**，和 `patch_rect` 的 `(x,y,w,h)` 一致）。
  单个数字仍然是方的，旧命令不受影响；`--latent-size` 可覆盖，默认 `W/8 × H/8`（512×384 → 64×48）。
- 两个维度**都必须是 8 的倍数**（VAE 按 /8 工作）。`--size 500x384` 会被直接拒绝并说明原因。
- **这一条以前是坏的，而且是静默坏的**：`app.py` 从来没把 `--size` 传给 `edit.py`，
  所以即使网页做成 4:3，模型也会把它压回方的。


第一版让用户提供"一张已经改好的参考图"，并按 mask 的 bbox 去**猜**内容该放哪。那是错的问题形状：
用户手上不是一个改好的整帧，而是**某个想要的东西的局部**——从任意照片截的一块、任意形状、任意大小。

第二版把 patch 做成可拖放，但 **mask 仍然画在源图上**。这产生了一个结构性错误，实测拍到了：
一张**大部分是黑色**的截图（人物 + 黑底）贴上去之后，
**黑底作为一个硬矩形被贴进了 Y**（"near-black pixels introduced = 18156"），
而 mask 和内容也可以对不上——两者是两套独立几何。

现在的一版，**mask 画在 patch 上，再由同一个放置动作带到源图上**：

1. 上传原图
2. 上传 **patch**（任意形状的局部）
3. **在 patch 面板上刷出物体本身**（`mark object`）——黑底不刷
4. `place / scale`：把放置框拖到位置上，用滑杆缩放
5. （可选）`touch up on source`：派生 mask 不满意时，**在源图上**补/抠
6. 填指令 → Run

**一个放置，一套几何。** `compose_patch()` 用同一个 `place_rect()` 同时决定
"内容贴到哪"和"mask 落在哪"，所以两者**不可能对不上**，
而且 **patch 自己没被圈到的背景永远不会进入 Y**。实测：

| | 派生 mask 占比 | 引入的近黑像素 |
|---|---|---|
| 有 patch mask（圈出物体） | **0.55%** | **0** |
| 无 patch mask（整个 patch 当内容） | 7.5% | **18156** ← 你拍到的那张 |

**patch 面板的比例 = patch 的比例**（长边 320px，4:3 → 320×240），patch **铺满**面板，
**不做 letterbox**。这不是外观问题：面板曾经是固定 320×320、patch 被 letterbox 塞在中间，
于是"圈物体的坐标系"（面板）和"放内容的坐标系"（patch）是两个不同的坐标系，
非正方形 patch 会漂移（实测 5–30 px，**正方形才恰好为 0**）。详见 §0.4 的 (8)。

#### 0.2.1 源图上的 touch-up 层（三态）

派生 mask 有两个它自己解决不了的情况：物体没被完全圈到（欠），
或者圈进了 patch 的一部分背景（过）。所以第 5 步给一层**三态**的手工修正：

| 画布上的像素 | 含义 | 预览颜色 |
|---|---|---|
| 透明（没画） | **不表态**，完全用派生的 mask | 红 = 派生 mask 亮着 |
| 白色（刷子） | **强制打开** | 绿 |
| 黑色（`erase` 打开后刷） | **强制挖洞** | 蓝 |

三个实现上的坑，都写进自检了：
- **必须走灰 128，不能靠透明。** 浏览器 canvas 的透明像素 RGB 是 `(0,0,0)`，
  而 `b64_png_to_array()` 会 `convert("RGB")`——**alpha 一丢，整张"不表态"就变成
  "全图强制挖洞"**。所以画布用透明（这样源图透出来能看见），
  发送前在 JS 里叠到 `#808080` 上，只有画过的像素才覆盖掉灰底。
- **"挖洞"和"本来就不在 mask 里"必须看得出区别**，否则用户看不见自己的擦除。
- **把唯一被圈到的物体整片挖掉 → 拒绝**（"the derived mask is empty"），
  而不是静默地跑一次什么都不改的编辑。

其他几个一起定下来的：
- **`fit=contain`（默认）**：patch 保持长宽比、居中放进放置框。绝不拉伸——
  拖出来的框和手打的 `(x,y,w,h)` 不保证同比例，而把一张脸拉宽不是任何人想要的。
- **`grow` / `feather`**：派生 mask 可以外扩、可以羽化（需要影子/光晕时用）。
- **不圈物体就直接拒绝**（"mark the object INSIDE the patch first"），
  而不是静默地贴一个含背景的方块。
- 每个请求落 `src.png` / `mask.png` / `derived_mask.png` / `y.png` / `out.png` /
  `y_meta.json`，出问题能回放。

### 0.3 远程机器上怎么打开

**先纠正一个可能的误解：启动 `app.py` 不会改变 JupyterLab 页面。** 它是一个**独立的服务器**，
UI 在**它打印的那个地址**上。启动后你看到的那段横幅就是它正常工作。

三条路，按"要不要依赖别人"排序：

#### 路 A ⭐ 用 SSH 端口转发 —— **不需要平台发布端口，不需要装任何东西**

你已经有 SSH 了（`ssh -p 20718 root@connect.bjb1.seetacloud.com`），所以**根本不用去找公网地址**：

```bash
# 在你自己电脑上开一条隧道（终端会停在这里，保持打开）
ssh -p 20718 -L 6006:127.0.0.1:6006 root@connect.bjb1.seetacloud.com
```

然后在**你自己电脑的浏览器**里打开：

```
http://127.0.0.1:6006/
```

远程`app.py` 照常跑（`--host 0.0.0.0 --port 6006`）。既然你能 SSH、又能在浏览器里用 JupyterLab，
这条路一定成立，而且**连 6006 是不是"发布端口"都不用管**——换个端口（比如 8765）一样通，
只要两边数字对上。启动时把 `--ssh` 传给它，横幅会直接打出这条命令：

```bash
python short_term/app.py --host 0.0.0.0 --port 6006 --mock \
    --ssh root@connect.bjb1.seetacloud.com:20718 \
    --lab-url "https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/lab/tree/RegionFlow/short_term"
```

#### 打开之后"正常"长什么样（对着这个核对）

**三栏横排**，从左到右：

1. **`1 - source`**：一个文件选择框，下面**一张黑的矩形**（上传后是原图），
   再下面是 `derived mask: empty`、`frame 512x384 (WxH, 1.33:1)`，以及一行按钮
   `touch up on source`，还有 **frame 选择器（默认 4:3）**。
2. **`2 - patch, then mark the object IN it`**：文件框，下面**一张 320×320 的方块**
   （上传后是 patch 图，周围一圈绿色边框），再下面按钮 `place / scale` / `mark object`。
3. **`3 - instruction & run`**：指令输入框、seed/steps/κ/κ_out、`MOCK` 勾选框、`Run edit` 按钮。

再下面一整行是 `log`。

**如果你看到的是"第 2 栏里三个方块并排"**，那是样式表写错了 id（0.4 的 (6)），
不是你的操作问题——**已经修了**，刷新页面即可。
**如果点 Run 只显示 `error` 或一条 `SyntaxError`**，那是 0.4 的 (4)/(5)/(7)，
也都修了；这类问题的共同点是**服务器端一切正常、图其实已经写到
`runs/st/ui/<时间戳>/` 了**，先 `ls runs/st/ui/*/` 看有没有东西。
**如果是 `edit.py exited 1 ...`**，看 0.4 的 (3)：原因现在直接写在报错里，
也存成 `<时间戳>/edit_log.txt`。

面板里的红/绿/蓝覆盖层分别是：**红=派生 mask**、**绿=你强制打开**、**蓝=你强制挖洞**。

```
  from your own machine, with SSH -- needs NOTHING published and nothing installed:
      ssh -p 20718 -L 6006:127.0.0.1:6006 root@connect.bjb1.seetacloud.com
      then open  http://127.0.0.1:6006/
```

#### 路 B 平台的发布端口（6006 / 6008）

平台只发布 6006 和 6008 两个 HTTP 端口。它们的公网地址在**控制台的实例页**
（「自定义服务」/「端口映射」那一栏）——不是 JupyterLab 那个 `:8443` 地址，
通常是另一条 `*.bjb1.seetacloud.com:<port>` 形式的链接，只能从控制台复制。

<small>想不靠控制台猜：在远程机器上 `env | grep -i -E 'port|host|url|instance'`，
有些镜像会把映射写进环境变量。</small>

#### 路 C 走 JupyterLab 的端口代理

```
https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/proxy/6006/
```

**但这条路要求装了 `jupyter-server-proxy`**（见下面的 404 说明），所以它排在 A/B 之后。

#### 路 D 完全不需要端口：`lab.run_line1`

```python
import sys; sys.path.insert(0, "/root/RegionFlow")
from short_term import lab
res = lab.run_line1(
    source="runs/st/real/src.png", patch="my_patch.png",
    rect=(160, 120, 200, 200),
    instruction="A hot air balloon should also be floating.",
)
```

构造和网页**完全相同**的 payload、调**同一个 `handle_run`**、在 cell 里把
source+mask / Y / result 三张图和指标一起画出来。放弃手拖 patch 与手画 mask。

`/proxy/` **不是 `jupyter_server` 自带的**，它由单独的 **`jupyter-server-proxy`** 包提供
（[官方文档](https://jupyter-server-proxy.readthedocs.io/en/stable/arbitrary-ports-hosts.html)）。
那个包没装，路由就根本不存在，服务器直接 404 —— 这正是这台机器上的情况。
我们的 app 对**非 API 路径一律返回页面**，所以 404 一定来自 Jupyter/平台，不是 app。

先诊断（notebook 一格）：

```python
import importlib.util
print(importlib.util.find_spec("jupyter_server_proxy"))   # None = 没装
```

三条路，按"代价从小到大"：

| 路 | 做什么 | 代价 |
|---|---|---|
| **1** | 用平台**发布端口 6006 的公网地址**（控制台「端口/服务」页），**绕开 Jupyter，什么都不用装** | 去控制台复制一个链接 |
| **2** | `pip install jupyter-server-proxy`，然后**重启 Jupyter 服务**，再开代理地址 | 一次安装 + 一次重启 |
| **3** | `lab.run_line1`，**完全不需要端口和代理** | 放弃手拖 patch / 手画 mask |

路 2 装完后**两个地址形式都可用**（`jupyter-server-proxy` 提供两种，只差 `absolute` 一段）：

```
https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/proxy/6006/
https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/proxy/absolute/6006/
```

前者会把前缀**改写掉**再转发，后者转发完整路径。**我们的路由按后缀匹配**（`.../api/run`），
所以两种都能用 —— 这也是为什么唯一的阻塞是"包没装"，而不是路由写错了。
启动横幅现在把**两个地址都打出来**。

⚠️ 路 2 的重启有风险：在云平台上 Jupyter 服务可能是平台托管的，杀掉它未必会自动重启。
所以**优先走路 1**。

**排错顺序**：代理 **403** ⇒ Jupyter 的 `allow_remote_access`（不是这个 app）；
代理 **404** ⇒ 包没装（见上）；两条都不通 ⇒ **路 3**。

**路 C：`lab.run_line1`（完全不需要端口）**

```python
import sys; sys.path.insert(0, "/root/RegionFlow")
from short_term import lab
res = lab.run_line1(
    source="runs/st/real/src.png",   # 也可以在 JupyterLab 文件浏览器里上传
    patch="my_patch.png",            # 任意形状的局部
    rect=(160, 120, 200, 200),       # 512 画布上的 (x, y, w, h)
    instruction="A hot air balloon should also be floating.",
)
```

构造和网页**完全相同**的 payload、调**同一个 `handle_run`**、在 cell 里把
source+mask / Y / result 三张图和指标一起画出来。放弃的是手拖 patch 与手画 mask
（位置用 4 个数字给，mask 默认等于 patch 矩形）。

**或者把 UI 嵌进 notebook**（daemon 线程起服务 + `IFrame`）：

```python
lab.embed("https://a1171371-wupv-7d353e12.bjb1.seetacloud.com:8443/jupyter/lab/tree/RegionFlow/short_term",
          port=8765, mock=False)
```

框里空白就单开标签页——嵌在 cell 里时画布拖拽在滚动区边缘会别扭。

### 0.4 已经踩过的坑（留个记录）

**(1) 代理 404**：`/proxy/` **不是 `jupyter_server` 自带的**，由单独的
**`jupyter-server-proxy`** 包提供
（[官方文档](https://jupyter-server-proxy.readthedocs.io/en/stable/arbitrary-ports-hosts.html)）。
包没装，路由就不存在，服务器直接 404。我们的 app 对**非 API 路径一律返回页面**，
所以 404 一定来自 Jupyter/平台，不是 app。诊断：

```python
import importlib.util
print(importlib.util.find_spec("jupyter_server_proxy"))   # None = 没装
```

装完后**两个地址形式都能用**（只差 `absolute` 一段）：

```
<base>/proxy/<port>/            前缀被改写掉
<base>/proxy/absolute/<port>/   转发完整路径
```

我们的路由**按后缀匹配**，所以两种都通——唯一的阻塞是"包没装"。

**(2) `edit.py exited 0 and no output was written`** —— 这条**是我的消息在撒谎**：
`exit 0` 意味着模型跑成功了，图已经写在磁盘上。

`--out` 一直是**词干（stem）而不是路径**：它会追加参数标签，这样一轮 sweep 不会互相覆盖
（`out_both_b5_kin0.6_kout1_flat.png`）。而 `app.py` 按固定名 `out.png` 去找，找不到，
就把一次成功的运行报成失败。修法两处：

- **`edit.py` 新增 `--out-exact`**：把 `--out` 当字面路径用，不加标签。给"只要一个文件"的
  调用方用；和 sweep 组合会静默覆盖，所以它**在 `--dry-run` 之前**就被拒绝（`_validate()`）。
- **`app.py` 改成 glob**（并显式传 `--out-exact`）：排除服务器自己写的
  `src.png/mask.png/y.png`，剩下的就是结果。报错信息现在**列出目录内容**而不是说"没写"。

**那次运行的图现在就在** `runs/st/ui/<时间戳>/` 里，名字带标签——`ls runs/st/ui/*/` 就能看到。

**(3) `edit.py exited 1 and no result PNG was found in runs/st/ui/...`**
—— 你拍到的这一条。**它当时有两层问题：一次真的运行失败，和一个把原因藏起来的报错。**

先说第二层，因为那让我（和你）多花了一个来回：报错**只写了退出码和目录列表**，
而 `edit.py` 早就把真正的原因打到 stderr 了——那些字在 `log` 面板里，
很容易被跳过。现在：

- **原因直接写进报错里**（stderr 尾部 1200 字符），不再只给一个退出码。
- 同时落到 **`<时间戳>/edit_log.txt`**（退出码 + 完整 argv + 完整 stdout/stderr），
  报错里带上这个路径。
- 失败响应也带 `rundir`，程序化调用方不用去解析字符串。

再说第一层：**`_REQUIRED` 是我改坏的。** 上一轮我为了让"复制不全"更容易发现，
把 edit.py 的必需文件列表从 4 个**扩到 9 个**（顺手把 `lab.py`/`app.py`/`score.py`/
`cache_item.py` 也加进去了）。但 edit.py **从来不 import 那几个文件**。结果：远程机器上
只要少一个 `lab.py`，**edit.py 就会拒绝运行**——模型、权重、输入全都没问题，它照样
`exit 1`，而且什么都不说（`SystemExit` 只有一行字，混在 stderr 里）。

现在分开：`_REQUIRED` 只列 **edit.py 真正 import 的四个**，
`_SHIPPED` 列全部文件**只给 `--doctor` 用**（缺了会标 `missing (optional)`，退出码仍是 0）。
自检**两个方向都测**：少了 `lab.py`/`app.py` 必须照样能编辑；少了 `sampler.py` 必须拒绝并点名。

**所以你那次运行的真正原因，需要看 log 面板或 `edit_log.txt`。** 最可能的两种：
① 我上面说的那个假失败（若你的 `short_term/` 少了 `lab.py` 或 `cache_item.py`）；
② `--guide both` 那条路需要 UltraEdit 的 diffusers fork（`StableDiffusion3InstructPix2PixPipeline`），
   而档 A/档 D 用的都是同一套权重——先确认 `python -c "import diffusers; print(diffusers.__version__)"`
   和 fork 的 pipeline 名字在远程机器上确实存在。

**(4) `SyntaxError: Unexpected token 'N', ..."ilution": NaN ... is not valid JSON`**
—— 也是你拍到的那条。**跑成功了，是响应读不出来。**

`score()` 真的会产出 `NaN`（没有背景可稀释时的稀释比）和 `inf`（对比度分母为 0），
而 Python 的 `json.dumps` **默认把 `NaN` / `Infinity` 写成裸 token**——这不是合法 JSON，
浏览器的 `JSON.parse` 直接抛异常，**整个响应丢掉**，包括已经算完的图和指标。
于是"完成"被显示成一条语法错误。修法三层：

- `json_safe()` 递归把非有限值换成 `null`，dump 时带 `allow_nan=False`。
- **`np.float32` 不是 `float` 的子类**（`np.float64` 才是），所以漏掉一次 `float()` 转换
  会让它变成"不可序列化的对象"→ `TypeError` → **同样丢掉整个响应**。现在
  `np.generic` / `np.ndarray` / `Path` 都归一化，最后一行兜底成字符串：**宁可类型难看，
  不能丢掉用户的结果。**
- dump 本身包了 try，绝不让一个编码问题变成没有响应体。

自检里有一条**真开 socket 的 POST**，用 `parse_constant` 抛异常来验证
（`json.loads` 默认接受裸 `NaN`，所以"文本里没有 NaN"既不对也会被 base64 误判）。

**(5) `render()` 被调用，但从没被定义** —— 修好 (4) 之后立刻撞上的下一个。
`$('run').onclick` 里写着 `if(j.ok)render(j);`，而页面里**没有这个函数**：
于是每次成功的运行都在 `try` 里抛 `ReferenceError`，被 `catch` 吃掉，
状态显示成光秃秃的 `error`，**图和指标一个都不显示**。现在补上了完整的结果面板
（三张图 + 指标表 + Y 列 + 判据），并加断言：`render` 必须存在，
且它读的每个响应字段都必须真的在响应里。

**(6) 样式表写的是已经不存在的 id** —— 你截图里"第二栏三个方块并排"就是这个。
规则里写着 `#mskc,#hl,#refc,#refsel`，这些 id 早就没了，所以 patch 面板
**完全没有定位**，三个 canvas 就并排铺成三个方块。
**写错 id 的 CSS 不报错，只是页面长得不对**——所以它需要一条测试：
现在自检会交叉核对"JS 里 `$('...')` 用到的 id"、"样式表里 `#...` 选中的 id"
和"HTML 里真实存在的 id"，任何一个对不上就失败。

**(7) 一个多余的大括号，整页 JS 全死** —— 我自己在写 (5) 的时候引进的。
JS 语法错误不会只坏一个函数：**整个 `<script>` 都不执行**，所有上传框、
按钮、画布全部失效，而服务器端**什么都看不出来**。
所以 `short_term/page_harness.js` 现在会把页面的 JS 抠出来，用 node
**真正解析并执行**（配一个 DOM stub），验证：能解析、加载不抛异常、
页面调用的函数都存在、三态 touch-up 层计数与编码正确、`render()` 对真实响应
（含被 sanitize 成 `null` 的 NaN）能出图。`app.py --self-test` 在有 node 时自动跑它，
没有 node 就跳过并打印一行说明——**不因为缺 node 就让整个自检失败**。

**(8) mask 相对贴上去的内容漂移（非正方形 patch）** —— 你拍到的"mask 有一点漂移"。

根因：**那个 320×320 的 patch 面板被当成 patch 本身用了。** 面板是固定的正方形，
patch 用 `patchFit()` **letterbox** 放在里面；而 `drawPlacement` 却把原始 patch
**拉伸**铺满整个方形面板。于是：

- 你圈物体是在**面板坐标系**里画的（也就是 letterbox 后的位置），
- 面板 1 里那个"已放置的 patch"预览却是**拉伸**坐标系，
- 服务器收到的是面板（带黑边），按**面板的方形比例**算 `place_rect`，而预览的
  `dstRect()` 按**patch 自己的比例**算 → 连 dst 框大小都不一样。

实测（源图像素，漂移 = 红 mask 位置 − 预览里角色位置）：

| patch | 比例 | 最大漂移 | 预览 dst 框 | 服务器 dst 框 |
|---|---|---|---|---|
| 400×300 | 1.333 | **30 px** | 240×180 | 180×180 |
| 400×380 | 1.053 | **5 px** ← 就是"一点" | 189×180 | 180×180 |
| 320×320 | 1.000 | **0 px** | — | — |

**正方形 patch 恰好是两者唯一相等的情形**——所以我之前所有测试都过了，这个 bug
一直藏着。

修法：**让面板的坐标系就是 patch 的坐标系**（和 frame 那次同一个套路）。
面板现在按 patch 的比例取长边 320（4:3 → 320×240），patch **铺满**面板，没有 letterbox，
于是面板→dst 是**纯缩放**，mask 和内容必然落在同一批像素上。
浏览器侧的约定由 `page_harness.js` 断言（面板比例 = patch 比例、dst 框比例 = 面板比例、
**非正方形** patch 也如此），服务器侧由 `propose_y` 自检断言（非正方形 patch 下
mask bbox 与 marker bbox 相差 ≤ 1 px），端到端由 `app.py` 自检断言。

**任意图片的四个实际限制**（都要知道）：

1. **方形**。`load_image` 会 resize 成 `size×size`，16:9 的照片会被**压扁**。
   做法：先裁成方形，编辑完再贴回去。
2. **mask 得你自己造**。我们到现在用的都是 MagicBrush 的真值 mask，
   **手画的或 SAM 的 mask 没测过**，边界质量可能有影响。
3. **只在 n=1 上测过**。`both` 追平真值这件事，需要 5–10 个 cache case 才算成立。
4. **"改对"的语义没验证过**。像素距离是方向不是判据；
   视觉确认算第四重证据，但"是不是一个热气球"仍需眼睛或 VLM 裁判。

---

外加 **`score.py`**：本机评分，**背景那一项用的是没被稀释的口径**，并且能用 `--expect`
判断"改对了方向"而不只是"改了"，用 `--tgt` 把**数据集的真值编辑**放在同一组坐标上当标尺。
做 β/κ 扫描之前先有它。

外加 **`cache_item.py`**：从 `/data/cache/<hash>` 里导出**真实 MagicBrush 三元组**
（`src.png` / `mask.png` / `tgt.png` / `meta.json`），替掉那张合成图。它只依赖 stock diffusers
的 `AutoencoderKL`，不需要 UltraEdit fork。

档 B（把 `z_src`/`mask` 接进 conditioning，训出自己的编辑器）见 `EDITOR_PLAN.md`。
顺序：**A → D → B → C**。

---

## 1. 怎么跑

```bash
# 同步自检：印出每个文件的大小和 sha256[:12]，以及它实际解析到的目录
python short_term/edit.py --doctor

# CPU 自检，秒级，不需要权重，不需要 GPU
python short_term/edit.py --self-test        # 参数装配（含 --guide both 的两个权重）
python short_term/guidance.py --self-test    # 引导代数（顺带交叉验证 sampler.py）
python short_term/backends.py --self-test    # pipeline 输入约定 + 注入层
python short_term/score.py --self-test       # 评分口径
python short_term/cache_item.py --self-test  # 真实三元组的导出
python short_term/propose_y.py --self-test   # patch 合成 + 三态 refine 层
python short_term/app.py --self-test         # HTTP/参数/代理路由/JSON + 页面 JS(需 node)
python short_term/lab.py --self-test         # notebook 路径 + proxy_url 推导

# 页面 JS 单独跑（app.py --self-test 已包含；没有 node 时用这条在别的机器上补）
node short_term/page_harness.js short_term/app.py

# 名字/编码检查：GBK 控制台下非 ASCII 的 .py 会炸，所以 .py 保持纯 ASCII
python tests/check_names.py $(ls short_term/*.py)

# 机制演示：MSE 的"中点"、噪声侧的分布、引导选模、以及两组对照
# 注意：本机实测 52s（含 torch import），冷启动可能超过 2 分钟，不是卡死
python short_term/demo_toy.py

# 管道校验，零依赖（内存里造图，不加载模型）
python short_term/edit.py --smoke

# ★ 换掉合成图：从 cache 导一个真实 MagicBrush 三元组（含真值 tgt.png）
python short_term/cache_item.py --cache /data/cache/<hash> --list 20
python short_term/cache_item.py --cache /data/cache/<hash> --index 137 \
    --out runs/st/real

# 档 A：先拿同口径基线（这一步不做，后面没有可比对象）
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --instruction "$(python -c "import json;print(json.load(open('runs/st/real/meta.json'))['instruction'])")" \
    --out runs/st/a.png

# 档 D：往参考图区域引导
python short_term/edit.py \
    --image synthetic_src.png --mask synthetic_mask.png \
    --guide-image synthetic_tgt.png \
    --instruction "change the red circle to blue" --guide target \
    --beta 5 --out runs/st/d.png

# 档 D：没有参考图，改成背景保护（pin_background 的软版本）
python short_term/edit.py \
    --image synthetic_src.png --mask synthetic_mask.png \
    --instruction "change the red circle to blue" \
    --guide source --invert-weight --sign +1 --kappa 0.3 \
    --out runs/st/protect.png
```

`--beta` / `--kappa` 接受列表，一次出多个结果，**seed 在整轮里固定**，所以每个设置看到的是
同一个初始噪声，比较隔离的是引导本身。

---

## 2. 约定（每一条都有历史教训）

**时间。** 全部用仓库的 `t`：**t=1 是噪声，t=0 是数据**，`z_t = (1−t)x0 + t·ε`，`v = ε − x0`。
PiGDM 的 `τ`（τ=1 是数据）只出现在从论文抄来的公式里，**τ = 1 − t**。两个都暴露，
`guidance.self_test` 断言它们一致。

**mask 极性。** **白 = 可编辑**，和 `bench/run_gate_c.py` 全部 mask 臂一致。
这个项目已经栽过一次极性 bug，训练和评测必须同约定。

**mask 交给 pipeline 时必须是 3 通道图像。** 它不是被阈值化的参数，而是**要被 VAE 编码的
条件图像**：`pipeline_stable_diffusion_3_instructpix2pix.py` 里
`mask_image_latents = self.prepare_image_latents(...)` → `self.vae.encode(image)`，
而 VAE encoder 的第一层卷积是 3 通道。传 `(H,W)` 会死在离调用点 20 层的 conv 里：

```
RuntimeError: Given groups=1, weight of size [128, 3, 3, 3],
expected input[1, 1, 512, 512] to have 3 channels, but got 1 channels instead
```

`bench/run_gate_c.py` 的 `cond_mask` 一直是这么做的（`Image.fromarray(...).convert("RGB")`，
白臂用 `Image.new("RGB", ...)`）。`short_term` 里是 `backends.to_pipeline_mask()`，**numpy 版、
同一件事**，并且 `self_test` 断言它与 `cond_mask` 的输出**逐字节相同**。
转换发生在 `SD3IP2PEditor.edit/edit_guided` 内部（唯一咽喉点），所以调用方不可能忘记；
`check_pipeline_inputs()` 会在进模型之前把单通道 mask 直接拒掉，而不是等 VAE 报错。

**引导权重 W 是单通道。** 它只走 `mask_to_latent()`，不进 pipeline。两个函数别混。

**文件同步。** `short_term/` 是手工同步的代码包，`__init__.py` 必须一起过去。
**没有 `__init__.py` 的目录在 Python 3.3+ 里仍然是一个 namespace package**，可以正常
`import short_term.backends` —— 所以少一个文件时报的是
`ModuleNotFoundError: No module named 'short_term.sampler'`，指向的是错的东西。
`edit.py` 和 `demo_toy.py` 现在都在导入前做一次同步检查，`--doctor` 会印出
**它实际解析到的目录**（这也是排查"机器上另有一份旧的 short_term 抢在前面"的办法）。

---

## 3. 已验证 / 未验证

| 组件 | 状态 | 证据 |
|---|---|---|
| `guidance.py` 全部代数 | **已验证（CPU）** | `--self-test`，见下 |
| `sampler.py` 循环 + 引导 + pin | **已验证（CPU）** | `demo_toy.py` 端到端 |
| `backends.py` 掩码/注入层 | **已验证（CPU）** | `--self-test`，含假调度器 |
| `score.py` 口径 | **已验证（CPU）** | `--self-test`，含真实输出的失败构造 |
| `app.py` 页面 JS（语法+运行） | **已验证（CPU）** | `page_harness.js`，见 0.4 (6) |
| `app.py` HTTP 端到端（mock） | **已验证（CPU）** | 自检里真开 socket 的 POST |
| `app.py` 真实浏览器交互 | **未验证** | 没人手点过：拖拽/画刷只有 DOM stub 级的证据 |
| `app.py` + 真模型（非 mock） | **未验证** | 只在 GPU 机器上跑过 `edit.py`，没跑过 UI |
| touch-up 层（源图上补/抠） | **已验证（CPU）** | 自检 + 真 socket：强制 ON 与挖洞都到了 `compose_patch` |
| diffusers 侧（真模型） | **已跑通（GPU）** | 档 A 与档 D 都出过图；见 §4 |
| 调度器是否 `sample+ds·v` | **有运行时检查** | `scheduler_dev`，见护栏 1 |

`guidance.py --self-test` 具体检查了什么（不是把公式重写一遍）：

- `(4)` 的 `r_τ²` **就是后验方差**：对 `a = τx₀+(1−τ)ε` 用**分箱**直接测 `Var(x₀|a)`
  （不假设它俩联合高斯），和公式对到 1e-3。
- 未加帽的增益 `((1−t)²+t²)/(t(1−t))` **在 t=0.5 取最小值 2**，且在 (0,1) 上 ≥2。
  **推论：β ≤ 2 时 min(β,·) 恒等于 β，整条"调度"退化成常数。**
- `(3)` 在线性路径上精确还原 `x₀`。
- 目标形式的修正**确实**把 `x₀_hat` 拉向 Y（并对 `sign=-1` 断言反方向）。
- `score_fn` 与目标形式**到 1e-10 一致**。
- `jacobian="vjp"` 对**闭式 Jacobian** 逐元素对；并断言它与 identity 形式**不相等**，
  所以测试不是空的。
- 直接混合形式在 `kappa_eq` 下**精确复现**速度形式（1e-9）。
- **`clean_blend(t, beta) = t·gain`，> 1 就是过冲**（见护栏 4）。
- 1-D 双模场：半区 mask 的引导能把模式翻过去。

### 五条护栏（都写在代码里，不是注释）

1. **注入点是 `scheduler.step`，不是 transformer。** 记录每步的
   `|solver_step − (sample + ds·v)|`（`scheduler_dev`）：对任何 `sample + ds·v` 形式的求解器
   都是 0。非 0 说明这套重加噪代数不描述该求解器，会报警而不是继续产出"自信地被误导"的图。
2. **对 `callback_on_step_end` 的依赖已去掉**：注入点搬到 `scheduler.step` 之后，
   档 D 不再需要 fork 支持那个 kwarg。
3. **shape 护栏**：`check_pipeline_inputs()` 在进模型前拒掉单通道 mask、以及 mask 与
   image 分辨率不一致的情况（不一致时 pipeline 会各自 resize，编辑区域就不在你以为的地方）。
4. **β 不是自由的旋钮：有效混合是 κ = t·gain。** 一步引导把干净估计移动
   `x0_new − Y = (x0 − Y)(1 − t·gain)`，所以 **κ > 1 就是过冲**（误差翻号并放大 |1−κ| 倍）。
   噪声端 t→1，因此 **β > 1 在第一步就已经过冲，β=30 是 30 倍过冲**。
   `guidance.clean_blend()` 把这条写成代码并断言。
   推论：**直接混合形式（`--kappa`）更安全**，它的混合是显式 κ 并被 `kappa_max` 夹住；
   速度形式的混合是 β·t，夹不住。
5. **`score.py` 的背景口径**：`d_out_content`（非白内容）而不是 `d_out_all`（整帧）。见 §6.1。

---

## 4. 在 GPU 机器上的执行顺序

**下面每一步都可以直接粘贴运行**（用仓库根目录里已有的合成图；换成你自己的图就改路径）。
`...` 不是 shell 语法 —— 早期版本用它当占位符，那是个文档 bug，已经改掉。

```bash
# 0. 不需要 GPU，不需要模型
python short_term/edit.py --doctor
python short_term/edit.py --self-test
python short_term/guidance.py --self-test
python short_term/backends.py --self-test
python short_term/score.py --self-test
python short_term/propose_y.py --self-test
python short_term/app.py --self-test
python short_term/lab.py --self-test
python short_term/demo_toy.py
python short_term/edit.py --smoke

# 1. 档 A 基线。**先做这个，否则后面没有同口径的可比对象。**
#    需要 UltraEdit 的 diffusers fork + GPU，并会下载检查点。
#    用真实三元组（cache_item.py 导出的），不要再用合成图。
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --instruction "<meta.json 里的 instruction 原文>" \
    --out runs/st/a.png

# 1b. **MASK CONTROL，必做，免费。** 同一张图、同一句指令，只把 mask 换成全白
#     （= Gate C 的 no-spatial-information 对照臂）。若它与 oracle mask 的结果
#     几乎一样，说明 mask 根本没到模型里 —— 那之前所有"模型不局部"的结论都要重读。
python short_term/edit.py \
    --image runs/st/real/src.png --mask white \
    --instruction "<instruction>" --out runs/st/a_white.png

# 2. 档 D：保护背景（无参考图）。**先扫 κ，它不会过冲。**
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --instruction "<instruction>" \
    --guide source --invert-weight --sign +1 \
    --kappa "0.3 0.6 1.0" --kappa-mode flat --out runs/st/protect_k.png

# 2b. **档 D 的 ORACLE 上界 —— 最有信息量的一次运行。**
#     把数据集真值当 Y、mask 当 W（不取反），即"把答案直接交给引导"。
#     若连真值都拉不动区域，这个机制无论 Y 是什么都做不到 WHAT；
#     若拉得动，那问题就变成"推理时拿不到 Y"——那是另一个结论，且正是档 B 用训练换的东西。
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --guide-image runs/st/real/tgt.png \
    --instruction "<instruction>" --guide target \
    --kappa "0.3 0.6 1.0" --kappa-mode flat --out runs/st/oracle_guided.png

# 2c. **两个项同时用。** inside→真值、outside→源图，各自一个强度。
#     这是"机制上界"：若它同时压住背景且 exp_gain > 0，机制就够用，缺的只是 Y。
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --guide-image runs/st/real/tgt.png \
    --instruction "<instruction>" --guide both \
    --kappa "0.3 0.6" --kappa-out "1.0" --kappa-mode flat \
    --out runs/st/both.png

# 3. 评分：模型行 vs **两条参考行**（no-op 与数据集真值）
python short_term/score.py --src runs/st/real/src.png \
    --mask runs/st/real/mask.png \
    --out "runs/st/protect_k_*.png" --tgt runs/st/real/tgt.png \
    --expect <指令要求的颜色> --rank

# 4. 有参考图时才用 --guide target
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --guide-image runs/st/real/tgt.png \
    --instruction "<instruction>" --guide target \
    --beta "0.3 0.5 1.0" --out runs/st/d_target.png

# 5. 上 Gate C 记分板
#    见 bench/run_gate_c.py；需要先有一个 gate-a 之外的自有臂
```

换成你自己的图就是把三条路径换掉。`--mask` 不给时默认 `white`（全部可编辑），
那是 Gate C 的 no-spatial-information 对照臂，**不是**"没有 mask"。

### 4.1 指令不能随便写（这一条是踩出来的）

第一版文档的示例指令是 `"make the shirt red"` —— 从仓库自己的经典编辑抄来的
（`region/alpha_field.py:8`、`bench/cases_refedit_final.json:5631`
"Let the shirt of the little girl be red."）。**指向那张两个圆盘的合成图，它是废话**，
而且代价是第一次实验的**编辑质量那一半全废**：mask 区域**本来就是红的**，所以正确答案是
"什么都不做"，`d_inside` 根本分不清"正确的 no-op"和"失败"。

合成图正确的指令是 **`"change the red circle to blue"`**。两条规则：

1. 指令必须指向**图里真实存在**的东西；
2. **mask 区域不能已经是**指令要求的那件事。

规则 2 由 `score.py --expect` 自动检查，会把 vacuous 的目标直接标出来，而不是给一个
无意义的 PASS：

```
$ python short_term/score.py --src synthetic_src.png --mask synthetic_mask.png \
      --out runs/st/a_none_b5_k0.png --expect red
  expect_gain     -2.562      (distance to the requested colour 0.124 -> 0.442)  AWAY/NO-OP
  !! --expect VACUOUS: the requested colour is already the nearest named colour to the masked
     region (distance 0.124 vs 0.188 to the next), so a correct answer would be 'no change' and
     d_inside cannot tell a correct no-op from a failure. Pick a target the region does not
     already have.
```

同一个输出换成有意义的靶子，`--expect blue` 给出 `expect_gain = −0.099`
（到蓝色的距离 0.621 → 0.682）—— **改错了方向**。这是 `target_changed PASS` 看不出来的东西。

---

## 5. 档 D 能做什么、不能做什么

**必须有一个 Y。** 引导是"把一步干净估计拉向某个东西"，所以 Y 要么是参考图（`--guide target`）、
要么是源图（`--guide source`，正号=保护、负号=逼着变），要么是一个**可微的分数**。

- **有参考图** ⇒ 档 D 立刻可用，而且这正是"把 A 区域改成 B 的样子"这类局部编辑。
- **只有指令 + mask，没有任何 Y** ⇒ **档 D 帮不上**。那是档 B（训练），不是引导。
- **分数分支**（`score_fn`，比如解码后用 CLIP 对指令打分）在**核心里实现了并与目标形式对齐**
  （1e-10），但 **CLI 还没有接**：它每步要 VAE decode，而且把感知家族那个**尺度问题**换个地方
  重现（梯度量级 vs 基底场量级）。要接的话先按仓库的口径量 `‖Δu‖/‖u‖`（`move_ratio` 那套）。

**已测到的档 D 行为。** 两套 fixture 都对同一个配置测过，结论一致但**真照片上的效果大得多**，
因为合成图的平坦白底本来就"改不动"，把效应整个藏住了。

真 MagicBrush 照片，指令 `"A hot air balloon should also be floating."`，mask 12.5%：

```
                                   d_in   d_out_c   contr  sat_in  exp_gain  held  valid
tier A（无引导）                   0.440   0.4474    0.98   0.281    -5.413  FAIL   FAIL
档 D protect, flat κ=0.3           0.437   0.0222   19.66   0.181    -5.363  FAIL   PASS
档 D protect, flat κ=0.6           0.424   0.0199   21.27   0.290    -5.182  PASS   PASS
档 D protect, flat κ=1.0           0.413   0.0196   21.03   0.317    -5.024  PASS   PASS
档 D protect, RAMP κ=0.3           0.433   0.3959    1.09   0.352    -5.311  FAIL   FAIL
(no-op)                            0.000   0.0000     nan   1.000     0.000  PASS   FAIL
(tgt: 数据集真值)                  0.070   0.0069   10.15   1.126     1.000  PASS   PASS
```

读法，四条一起：

1. **WHERE 拿得到，零训练。** 背景变化 0.4474 → **0.0196**（23×），contrast 0.98 → 21。
   离数据集自己的背景保持（0.0069）只差 2.9×。
2. **`d_in` 保持在 0.41 以上**，远高于 0.06 —— **保护不是靠冻结画面买来的**。
3. **WHAT 拿不到。** `exp_gain` 从 −5.41 到 **−5.02**，真值是 **+1.000**。
   而且 `sat_in` 0.317 vs 真值 **1.126** —— 区域仍然被漂白。
   这正是构造上的预期：W = 1−mask 在 mask 内**恒为 0**，保护项碰不到 WHAT。
4. **`contrast` 是比值，会替模型说好话。** 我们 contrast 21.27 高于真值 10.15，
   但那是因为我们的 `d_in` 是真值的 6 倍。**要成对报绝对值**，不能只报比值。

**`--kappa-mode` 默认已是 `flat`。** 同一名义 κ=0.3，ramp 的 `d_out_c` 是 0.3959（contrast 1.09，
等于没局部化），flat 是 0.0222（contrast 19.66）—— **18×**。合成 fixture 上两者是
0.0243 vs 0.0255，看不出差别。**ramp 是有害的默认值，已经改掉**，文件名里现在始终带 mode。

**一个未解决的副作用。** `sat_in` 随 κ 变化的**方向**在两套 fixture 上相反：
合成图上 κ 越高 sat_in 越差（0.546→0.339），真照片上反而越好（0.181→0.317）。
原因是潜变量在空间上耦合、引导项虽在 mask 内为 0 但会改变整条轨迹，
而副作用的方向取决于源图 mask **外面**长什么样（合成图外面是平坦白色）。
所以"保护外面"对 mask 内的影响不是零，符号随 case 变——**不能假定它是无害的**。

### 5.1 两个引导项修的是**相反的半边**，单独用哪个都不够

把真值直接当 Y 交给引导（`--guide target --guide-image tgt.png`，W = mask，作用在 mask 内）：

```
                                   d_in   d_out_c   contr  sat_in  exp_gain  held  valid
tier A（无引导）                   0.440   0.4474    0.98   0.281    -5.413  FAIL   FAIL
档 D protect（source, 1−mask）     0.424   0.0199   21.27   0.290    -5.182  PASS   PASS
档 D oracle（target, mask）        0.097   0.4444    0.22   0.926    +0.322  FAIL   FAIL
(tgt: 数据集真值)                  0.070   0.0069   10.15   1.126     1.000  PASS   PASS
```

**这是全项目最干净的一张对照表。** 两个项各修一半，而且互不干扰：

| 项 | W | 作用在 | 结果 |
|---|---|---|---|
| `source` | `1−mask` | 外面 | **背景修好了**（0.447→0.020），编辑仍然错 |
| `target` | `mask` | 里面 | **编辑修好了**（`sat_in` 0.281→**0.926**，`d_in` 0.440→**0.097** 对真值 0.070），背景仍然毁（0.444） |

几点必须说清：

1. **`exp_gain` 从 −5.41 变成 +0.322，其中"变好"有一部分是构造性的**
   （把 Y 拉向 tgt，到 tgt 的距离当然会降）。**但三件事不是构造性的**：
   ① 它只到 **+0.322 而不是 +0.9** —— 手里有答案也只拉得动三分之一，说明引导力相对于基底场是**有界**的；
   ② `sat_in` 0.281 → **0.926**（真值 1.126）——**漂白被逆转了**，输出成了一个像样的区域，不是涂抹；
   ③ `d_in` 落到 **0.097**，与真值的 **0.070** 只差 1.4× —— **改动量对了**。
2. **κ 在 oracle 里几乎不起作用**（0.3→0.6→1.0 给 0.242→0.314→0.322），已饱和。
3. **两个项正交**，所以"同时用两个"是显然的下一步 —— 那正是 `--guide both`。

### 5.2 `--guide both`：两个项同时用

```bash
python short_term/edit.py \
    --image runs/st/real/src.png --mask runs/st/real/mask.png \
    --guide-image runs/st/real/tgt.png \
    --instruction "<instruction>" --guide both \
    --kappa "0.6" --kappa-out "1.0" --kappa-mode flat \
    --out runs/st/both.png
```

- `inside`：W = mask，Y = `--guide-image`，强度 = `--kappa`
- `outside`：W = 1−mask，Y = 源图，强度 = `--kappa-out`（默认同 `--kappa`）

**两者可以分开调**，因为它们饱和点不同（保护在 0.6 就平了，里面的拉还在动）。
文件名现在是 `..._b5_kin0.6_kout1_flat.png`，两个强度都在名字里。

**必须提前说清的边界**：`--guide both` 需要**真值图**，而推理时你没有。
所以它的意义**不是"这就是产品"，而是"给机制一个上界"**——而这个上界**已经测到了**，见下。

### 5.3 结果：`both` 几乎追平数据集真值，而且两个项是**互相加强**的

同一张真 MagicBrush 照片、同一句指令、同一个 mask、同一 seed：

```
                                      d_in   d_out_c   contr  sat_in  exp_gain  held  valid
tier A（无引导）                      0.440   0.4474    0.98   0.281    -5.413  FAIL   FAIL
档 D protect only（source, 1−mask）   0.424   0.0199   21.27   0.290    -5.182  PASS   PASS
档 D oracle only（target, mask）      0.097   0.4444    0.22   0.926    +0.322  FAIL   FAIL
★ 档 D BOTH（in κ=0.6 / out κ=1.0）  0.075   0.0166    4.54   1.053    +0.788  PASS   FAIL
(tgt: 数据集真值)                     0.070   0.0069   10.15   1.126     1.000  PASS   PASS
```

**`both` 在每一个实质轴上几乎追平真值：**

| 轴 | both | 真值 | 比值 |
|---|---|---|---|
| 改动量 `d_in` | **0.075** | 0.070 | **1.07×** |
| 背景 `d_out_c` | **0.0166** | 0.0069 | 2.4× |
| 饱和度 `sat_in` | **1.053** | 1.126 | 0.94× |
| `exp_gain` | **+0.788** | 1.000 | — |
| `held` | **PASS** | PASS | — |

**三件不是构造性的事**（`exp_gain` 变好有一部分是把 Y 拉向真值的同义反复，这三件不是）：

1. **`d_in` 与真值差 7%** —— 改动量对了，而 tier A/oracle-only 分别是 6.3×/1.4×。
2. **`sat_in` 1.053 vs 真值 1.126** —— 漂白被逆转，输出是像样的区域。
3. **两个项互相加强，不只是兼容。** `both` 在 `d_in`（0.075 vs 0.097）**和** `exp_gain`
   （0.788 vs 0.322）上都**优于 oracle-only**。原因：潜变量空间耦合，把 mask **外面**
   钉向源图会消掉"整帧重生成"的漂移，于是里面那一拉不用再跟漂移对抗。
   **保护项不只是保护，它同时改善了编辑本身。** 这条对档 B 的设计有直接影响。

κ 影响很小（0.3 与 0.6 给 0.763/0.788），所以这个配置对 κ 不敏感、很稳。

### 5.4 `valid` 那一列的 FAIL 是判据问题，不是模型问题

`valid` 是 `d_in > 10·d_out_all`，这里 `d_out_all ≈ d_out_c = 0.0166`，于是要 `d_in > 0.166`，
而 both 的 0.075 达不到。**但真值自己只有 10.15 的比值 —— 只比阈值 10 高 1.4%。**
`target_changed` 更紧：真值 `d_in` 0.070 只比 0.06 高 1.17×。

**一个连数据自己都只勉强过的判据，分不出好run和坏run。** 评分器现在会**自动检测并报警**：

```
!! THE GROUND TRUTH BARELY CLEARS target_changed (1.17x).
   A criterion the data itself clears by less than 1.5x cannot discriminate a
   good run from a bad one. Do NOT read those columns as verdicts on this case
   -- compare against the tgt row instead.
```

并改报**「相对真值的倍数」**（这一列才是校准过的）：

```
against the ground truth (multiples; 1.0 = as good as the dataset's own edit):
  run                                 d_in/tgt  d_out_c/tgt  sat_in/tgt
  both_..._kin0.6_kout1_flat.png          1.07         2.40        0.94
```

这条也钉进了自检（用这次真实的 ballooon 数字）：`valid_wholeframe` 余量 1.01×、
`target_changed` 余量 1.17×，两个必须被列为 weak。

### 5.5 决策点已经解决：**机制够用，缺的是 Y**

预注册的判据是「`both` 同时做到 `d_out_c` < 0.02 **且** `exp_gain` > 0 ⇒ 机制够用」。
**两条都成立**（0.0166 与 +0.788）。所以：

- **现在就有可交付物：参考引导的局部编辑，零训练。** 给源图 + mask + 一张"这块该长成什么样"
  的参考图，它能做到上面那一行。这是真的能力，不是演示。
- **缺的是指令版**：推理时没有 Y。**档 B 的目标因此被重新定义** —— 不是"训出一个好编辑器"，
  而是"**给定 (源图, 指令, mask) 预测出 Y**"，好到引导落在 oracle-Y 落的地方。
  目标可量化，就是上面 `both` 那一行：`d_in` ≈ 0.075、`d_out_c` < 0.02、
  `sat_in` ≈ 1.05、`exp_gain` > 0.7。

**边界（必须一起说）**：`exp_gain` 有构造性成分（拉向 Y ⇒ 到 Y 的距离下降）。
所以判"成不成"要靠**不构造的三条**：`d_in` 与真值的比、`sat_in`、以及 `--tgt` 那一行。
另外**像素距离 ≠ 语义**——用户视觉确认这两张"可以接受"，但"是不是一个热气球"仍需眼睛或
Gate C 的 VLM 裁判。

### 5.6 多 case 复核与参考图敏感性 —— 下一步的两个实验

`both` 只在**一个** case 上追平了真值。两件事必须做，都便宜：

**实验 1：5–10 个 cache case 上的分布。** 决定 `both` 是成立还是侥幸。

```bash
# 挑 case（mask_frac 在 [0.02,0.30]、改动量最大的一批）
python short_term/cache_item.py --cache /data/cache/<hash> --list 12

# 逐个导出、跑、评分；指令从 meta.json 取，不用手打
for I in 137 200 251 302 355; do
  python short_term/cache_item.py --cache /data/cache/<hash> --index $I \
      --out runs/st/c$I >/dev/null
  INS=$(python -c "import json;print(json.load(open('runs/st/c$I/meta.json'))['instruction'])")
  python short_term/edit.py --image runs/st/c$I/src.png --mask runs/st/c$I/mask.png \
      --guide-image runs/st/c$I/tgt.png --instruction "$INS" \
      --guide both --kappa 0.6 --kappa-out 1.0 --kappa-mode flat \
      --out runs/st/c$I/both.png >/dev/null
  python short_term/edit.py --image runs/st/c$I/src.png --mask runs/st/c$I/mask.png \
      --instruction "$INS" --guide none --out runs/st/c$I/base.png >/dev/null
done
```

然后逐 case 评分（`--rank` 会把该 case 的模型行和真值行放在一起），
**要看的是 `d_in/tgt` 与 `sat_in/tgt` 的分布**，不是某一个 case 的绝对值。

**实验 2：参考图能有多差。** 决定"生成式 Y"（来源③）是否可行 —— 这是最关键的未知数。
把真值 Y 故意劣化，看 `exp_gain` 掉多快：

```bash
python - <<'PY'
import numpy as np, sys
from PIL import Image, ImageFilter
from pathlib import Path
src = Path("runs/st/c137")
tgt = Image.open(src/"tgt.png")
tgt.filter(ImageFilter.GaussianBlur(4)).save(src/"tgt_blur4.png")   # 模糊
tgt.filter(ImageFilter.GaussianBlur(16)).save(src/"tgt_blur16.png")
a = np.asarray(tgt).astype(np.float32)
rng = np.random.default_rng(0)
for s, name in ((0.05,"tgt_noise05.png"), (0.15,"tgt_noise15.png")):
    n = a + rng.normal(0, s*255, a.shape)
    Image.fromarray(np.clip(n,0,255).astype("uint8")).save(src/name)
for dx, name in ((8,"tgt_shift8.png"), (24,"tgt_shift24.png")):    # 错位
    Image.fromarray(np.roll(a, dx, axis=1).astype("uint8")).save(src/name)
Image.open(src/"src.png").save(src/"tgt_wrong.png")                # 最极端：拿源图当 Y
print("variants written")
PY
```

**预注册判读**：若 `exp_gain` 在 `blur4`/`shift8` 上只掉一点点（比如仍 > 0.6），
那"生成一个粗略对位的目标内容"就够用，来源③可行，**指令版不用训练就能拼出来**。
若一劣化就崩，那 Y 的质量必须很高，只能靠训练（档 B）。

**实验 3（更便宜，先做也行）：只给 mask、不给参考，看 baseline 在哪里。**
`--guide both` 但把 `--guide-image` 指向源图自己 —— 那等价于"区域内保护 + 区域外保护"，
应当几乎不产生编辑。这确认 `--guide-image` 确实是唯一的"WHAT 来源"。

---

## 6. 要打败的数字

151 例 Gate C，先写下来再看结果：

| 臂 | 数 | 出处 |
|---|---|---|
| `none`（RefEdit-SD3） | mis-targeted **28%** | `GATE_C_REPORT §3` |
| `white`（UltraEdit + 全白 mask） | mis-targeted **30%** | 同上 |
| `freeform` | mis-targeted **40%** | 同上 |
| **`oracle`**（UltraEdit + oracle mask） | mis-targeted **0%**，clean **22%**，VLM Overall median **4.87** | 同上 §3/§5 |
| `sibling`（同类别其他实例） | mis-targeted **94%** | 同上 §3 |
| pairflow σ0=0.0 | valid(K=10) 38/151 | `GateA.md §12` |

**"能用"的定义：oracle mask 下 clean 明确高于 22%，或 无 mask 下 mis-targeted 明确低于 28%。**
不要单看 `valid`，它的 K 依赖会交叉（oracle 16 vs none 38 at K=10；128 vs 92 at K=5）。

### 6.1 两个本机口径，必须一起读

**`d_out_content`，不是 `d_out_all`。** 同一次输出：

```
d_out_all      0.0513   <- 整帧减 mask，被 21 万像素的平坦白底稀释
d_out_content  0.5789   <- 非白内容：绿圆被抹平（饱和度 0.588 -> 0.176）
```

**差 11.3 倍。** 平坦白底重建出来还是平坦白底，它不可能变，所以它在分母里只能稀释信号。
`bench/off_instance.py` 在基准尺度上就是为这个存在的（"RefEdit-SD3 regenerates the whole
frame"）；`score.py` 是本机尺度上的同一个东西。
更进一步：**Gate C 自己的 `valid = d_inside > 10·d_out_all` 也会被稀释骗过去**——
`score.py --self-test` 里构造了一个案例，`valid` 是 PASS 而画面里唯一另一个物体已经被毁。

**`expect_gain`，不是 `d_inside`。** `d_inside > 0.06` 只说明"发生了改变"。
`--expect` 给出到期望内容的距离变化：> 0 朝目标、~0 没动、< 0 **改反了**。
一个运行可以背景保护得完美而 `expect_gain < 0`，那读作"保护住了，但没做这个编辑"。

`--expect` 有三种形式：

| 形式 | 用在 | 锚点 |
|---|---|---|
| `--expect blue` / `--expect 0,0,255` | 颜色/属性编辑 | no-op = 0 |
| `--expect tgt`（需 `--tgt`） | **任何**编辑，包括"加一个热气球"这种没有颜色的 | no-op = **0.000**，精确复现真值 = **1.000** |

`--expect tgt` 是通用形式：期望内容就是**数据集自己的编辑结果**在 mask 内的那一块。
没有颜色的指令（`"A hot air balloon should also be floating."`）只能用它——
纯颜色名在那里没有东西可指，评分器就只能说"区域变了"，说不出"变对了没有"。
若真值本身在该 mask 内几乎等于源图，它会直接报警 `VACUOUS`（那一格没有编辑可测）。

**注意像素距离 ≠ 语义。** 两个不同的热气球在像素上都算"靠近真值"。
`expect_gain` 是个**方向性**的量，`tgt` = 1.000 是上界、no-op = 0 是零点；
"到底像不像热气球"要眼睛或 Gate C 的 VLM 裁判，本地做不到。

**`--tgt`：把真值编辑放在同一组坐标上。** 这是 `STAGE1_REPORT §3` 明文要求的那条协议
——"**先用你的度量给真值打分再相信它；如果真值排不到前面，那就是度量错了**"。
在合成 fixture 上的实测：

```
output                              guide  beta  kappa    d_in  d_out_c  dilut  contr  sat_in exp_gain  held  valid
st_a_none_b5_k0.png                     -     -      -   0.409   0.5789   11.3   0.71   0.483   -0.099  FAIL   FAIL
(no-op: the source itself)              -     -      -   0.000   0.0000    nan    nan   1.000    0.000  PASS   FAIL
(tgt: the dataset's real edit)          -     -      -   0.562   0.0000    nan    inf   1.000    0.695  PASS   PASS
```

真值得 **+0.695**、背景 **0.0000**、`contrast = inf`；模型得 **−0.099**。
所以**度量没有坏，是模型不行** —— 这句话必须由这三行一起支撑，不能由模型那一行单独说。
若真值那行也过不了 `d_out_content < 0.02` 或 `expect_gain ≤ 0`，评分器会直接警告：
**那是阈值比数据还严，要把它当刻度报告而不是判据。**

**别用眼睛判。** Stage 0 有一次自动化"窗口"判决，三条全错，看一眼图几秒钟就能推翻
（`STAGE0_FINDINGS §1`）。图是证据的一部分，不是判据。

---

## 7. 下一步

- **档 B**：`EDITOR_PLAN.md §2 档 B`。核心一句：`scripts/gate_b_meanflow.py:452-464` 的训练目标
  已经是对的（`x0 = z_tgt`，`v = ε − x0`），同一 batch 里 `z_src`/`mask` 已经躺着，
  **缺的只是把这两个张量接进 conditioning**（`in_channels=48` 的 UltraEdit 先例）。
  **条件路径已经建好并 CPU 全测过**：`transport/source_cond.py` + `--source-cond`
  （16→48 手术、预训练列逐位不变、新增列零初始化→起点等于 base 模型、**并且断言新列真拿得到
  梯度**——零初始化 + 冻结的 base 会得到一个永远学不到东西的条件路径，这是计划里没写的坑）。
  **没跑的**：真实训练、以及 §3 要求的同口径自身基线（都要 GPU）。命令和验收标准见
  `EDITOR_PLAN.md §2 档 B 的实现状态`。
- **档 C**：档 B 出来之后，加实例级选择性，用 `sibling` 臂（94% 误靶）打分。
- **回到 Gate A 只作为负结果**，不要再投训练。唯一值得留的便宜实验见 `EDITOR_PLAN.md §6`。
