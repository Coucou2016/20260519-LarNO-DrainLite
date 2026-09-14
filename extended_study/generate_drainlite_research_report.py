from __future__ import annotations

import base64
import csv
import html
import os
import re
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper_draft" / "drainlite"
FIG_DIR = PAPER_DIR / "figures"
TABLE_DIR = PAPER_DIR / "tables"
OUT_HTML = PAPER_DIR / "report.html"
OUT_MD = PAPER_DIR / "report.md"
OUT_PDF = PAPER_DIR / "report.pdf"
ROOT_HTML = ROOT / "report.html"
ROOT_MD = ROOT / "report.md"
ROOT_PDF = ROOT / "report.pdf"


FIGURES = [
    {
        "id": "fig01",
        "file": "fig01_framework.png",
        "title": "LarNO-DrainLite 轻量排水残差修正总体框架",
        "short": "本图说明本研究不是重训完整 LarNO 主干，而是在已有地表洪涝结果之后叠加一个轻量、可解释的排水残差修正层。",
        "detail": [
            "这张图放在报告开头，是为了先回答“这项工作到底改了哪里”。左侧对应原始城市雨洪预报链条：降雨、地形和已有模型给出无管网或弱管网表达的地表水深。中间加入道路对齐的概化管网特征，包括雨水口、排放口、管线、管径、坡度、满流能力、覆土深度和到排放口距离。右侧是 DrainLite（轻量排水残差修正模型），它学习的不是完整水动力过程，而是“加入概念性排水入口后，水深应当削减多少”。",
            "读图时可以把箭头理解成信息流。降雨和数字高程模型（Digital Elevation Model，数字高程模型，表示每个栅格的地面高度，是地表径流沿坡度流动和低洼处积水的基础控制量）决定水往哪里汇集；管网特征告诉模型哪里更可能被排走；残差学习器把这两部分信息合起来，输出非负削减量。这里的 residual（残差，在本研究中不是误差随便相减，而是无管网地表水深与带排水入口水深之间可由排水解释的差值）对应公式中的 h_surface minus h_sink。它的物理意义是“排水系统从地表移走的等效水深”。",
            "图中的关键约束是 h_drainlite <= h_surface。通俗说，当前版本只允许管网把水排掉，不允许管网反过来向地表冒水。这与真实城市管网可能发生的 surcharge（管网壅水或冒溢，指管道满流后水从检查井返回地表）不同，因此本图也界定了方法边界：它是 sink-only（只考虑排水入口移除水量的源汇项）轻量模型，不是完整双向地表-管网耦合模型。"
        ],
    },
    {
        "id": "fig02",
        "file": "fig02_dem_network.png",
        "title": "研究区域数字高程模型与概化管网叠加图",
        "short": "本图用于检查地形、路网和管网是否在同一空间位置上对齐，是后续所有模型结果可信的前置条件。",
        "detail": [
            "这张图的背景是前期反复检查过的“管网和数字高程模型相对位置问题”。城市洪涝模拟对空间对齐非常敏感，若管网整体偏移、上下翻转或与道路错位，模型可能会在错误位置排水，进而把削峰效果解释到不存在的地方。因此本图在报告中承担质量控制作用。",
            "读图时，底图的颜色表示地面高程。颜色较低的区域通常更容易积水，因为二维地表水动力方程中的重力项会推动水沿高程梯度向低处汇集；颜色较高的区域更像分水岭或阻水边界。叠加的线状要素表示概化管线，点状要素表示雨水口、检查井或排放口等节点。若管线大体沿着平坦道路或建筑间空地延伸，说明“道路对齐管网先验”在空间上是合理的。",
            "从这张图可以得到两个结论。第一，报告采用的是当前已经校正后的相对位置，而不是早期误判时上下翻转的版本。第二，管网仍然是概化管网，不是官方精细排水资产。也就是说，它适合研究“管网位置先验能否帮助学习排水削峰”，但不能直接用于工程设计或真实排水能力校核。"
        ],
    },
    {
        "id": "fig03",
        "file": "fig03_data_flow.png",
        "title": "数据、物理标签与机器学习训练流程",
        "short": "本图把 MIKE reference、ITZI surface-only、ITZI + sink、SWMM standalone 与 DrainLite 的关系拆开，避免把不同物理含义的结果混在一起。",
        "detail": [
            "这张流程图是整篇报告最重要的“概念防混淆图”。MIKE Plus（商业一维/二维城市水动力模型，原 LarNO 论文中用作参考结果）在这里是外部参考；ITZI surface-only（ITZI 地表二维动力学求解结果，不调用概念性排水入口）是 DrainLite 的基底；ITZI + sink（在地表求解中加入道路对齐概念性排水入口源汇项）是 DrainLite 的主监督标签；SWMM standalone（Storm Water Management Model，美国环境保护署雨洪管理模型的独立一维管网动力波计算）只提供事件级管网指标。",
            "读图时要注意箭头是否进入训练目标。DrainLite 的训练目标来自 ITZI + sink 与 ITZI surface-only 的差值，而不是直接来自 MIKE reference。这样做的原因是：MIKE reference 可能包含其自身的真实或参数化管网体系，而本研究的概化管网是另一个受控假设。如果直接把概化管网特征拿去拟合 MIKE，容易把“真实参考差异”和“合成管网削峰”混为一谈。",
            "图中的 SWMM standalone 分支也要谨慎理解。SWMM 的 dynamic-wave（动力波方法，完整求解一维 Saint-Venant 方程的惯性、压力和摩阻项，用于描述管网内非恒定流）结果能告诉我们管网系统在事件尺度上有多少入流、出流和储水，但当前版本没有在线接收 ITZI 地表水深，也没有把管网壅水回灌到地表。因此它不是完整耦合标签，只是辅助解释和特征对照。"
        ],
    },
    {
        "id": "fig04",
        "file": "fig04_event75_peak_maps.png",
        "title": "测试事件 event75 的峰值水深空间对比",
        "short": "本图展示一个具体事件中 MIKE、ITZI、ITZI + sink、DrainLite 与误差/削减量的空间差异。",
        "detail": [
            "这张图用 event75 做单事件剖面，是为了让读者在看汇总指标之前先建立空间直觉。峰值水深图不是某一个固定时刻的截图，而是每个栅格在 6 小时模拟过程中的最大水深，因此它强调“哪里曾经最深”。",
            "读热图时，颜色越深通常表示最大水深越大。MIKE reference 子图代表原 LarNO 数据中的外部参考；ITZI surface-only 子图代表不考虑概念性排水入口时地表动力学模型的结果；ITZI + sink 子图代表加入道路排水入口后的受控标签；DrainLite 子图代表模型从 surface-only 结果中预测削减量后得到的水深。误差图通常以正负色带表达，如果某处误差为正，说明 DrainLite 相对标签偏高；为负则说明偏低。削减量图显示模型认为管网从地表移走的等效水深。",
            "从 event75 可以看出，DrainLite 对大范围浅水区的修正更稳定，但对极端局部深水峰值仍然存在不足。这与后面的峰值误差表相呼应：平均绝对误差能显著下降，但峰值点误差仍偏大。原因并不神秘，树模型在逐像元样本上学习的是常见空间模式，而极端峰值往往受非常局部的边界、低洼坑、网格连接和水动力汇流控制，单靠静态管网特征难以完全恢复。"
        ],
    },
    {
        "id": "fig05",
        "file": "fig05_ablation.png",
        "title": "DrainLite 消融实验结果",
        "short": "本图比较 surface-only、base、mask-only、hydraulic-only、all-static 与 SWMM-assisted 多组模型，回答管网特征到底有没有用。",
        "detail": [
            "消融实验（ablation study，逐步移除或替换特征组以判断各部分贡献的方法）是这项轻量创新能否站住的核心证据。它不只是展示一个最好模型，而是问：如果不用管网特征，仅靠地形和降雨能做到什么程度？如果只用管网位置，不用管径坡度等水力参数又会怎样？如果加入 SWMM 事件级指标是否进一步提高？",
            "图中的每个柱或点对应一个模型变体。surface-only 是没有任何残差修正的地表模型；base 是有残差学习但没有管网特征；mask-only 加入雨水口、排放口、管线掩膜和距离排放口等空间位置先验；hydraulic-only 加入管径、坡度、能力和覆土深度等水力属性；all-static 使用全部静态管网特征；SWMM-assisted 在 all-static 基础上再加入 SWMM standalone 的事件级汇总指标。",
            "最重要的读法是比较误差越低越好、临界成功指数越高越好。结果显示 all-static 相对 surface-only 的平均绝对误差下降约 66.7%，相对 base 下降约 18.2%。这说明排水信息不是简单由地形和已有水深间接推出来的，显式管网特征确实提供了额外信号。同时 mask-only 与 all-static 很接近，说明在当前概化网络和轻量模型下，管网“在哪里”比管径、坡度等参数“取多少值”更有解释力。"
        ],
    },
    {
        "id": "fig06",
        "file": "fig06_feature_importance.png",
        "title": "DrainLite 特征重要性分析",
        "short": "本图解释模型主要依赖哪些输入来判断排水削减量。",
        "detail": [
            "特征重要性图的目的，是把机器学习模型从黑箱往可解释方向拉一步。这里的 feature importance（特征重要性，指模型在分裂或预测中对某个变量的依赖程度）不是物理定律本身，但可以帮助判断模型是否学到了符合常识的关系。",
            "读图时，横轴通常表示重要性大小，纵轴列出变量名称。靠前的变量说明模型在预测削减量时更频繁或更有效地使用它们。比如 surface depth（地表水深，代表某格是否有水可排，是削减量的上限来源）、pipe mask（管线掩膜，表示该格或邻域是否接近管网）、inlet density（雨水口密度，代表附近可进入管网的机会）和 distance to outfall（到排放口距离，反映排水路径远近）通常应当具有较高解释力。",
            "从图中可以读出一个比较稳健的结论：模型并不是只在记忆事件编号，也不是只按降雨强弱做全域统一削减，而是在使用空间管网位置和局部水深来决定哪里削得更多。这正是本研究想表达的“道路对齐管网先验”价值。不过，特征重要性不能证明真实物理因果；它只说明在当前 ITZI + sink 标签和训练样本中，这些变量对拟合残差最有用。"
        ],
    },
    {
        "id": "fig07",
        "file": "fig07_swmm_summary.png",
        "title": "SWMM standalone 管网指标汇总",
        "short": "本图展示独立一维管网模型在各事件中的入流、出流、储水和连续性误差，用来判断 SWMM 分支是否数值上可用。",
        "detail": [
            "SWMM standalone 分支的作用不是直接生成地表水深标签，而是为概化管网提供事件级水力背景。图中的 routing inflow（管网路由入流，表示进入一维管网系统的总体水量）、outfall volume（排放口出流体积，表示最终从管网排出的水量）、routing storage（管网储水量，表示事件末管道和节点内部仍保存的水量）和 continuity error（连续性误差，表示数值计算中水量守恒偏差）共同描述了管网计算是否合理。",
            "读图时，首先看连续性误差是否过大。若连续性误差很高，说明一维管网求解的水量平衡不好，后续把它当作辅助特征就有风险。本研究中平均连续性误差约 1.24%，说明 standalone 分支在数值水量平衡上基本可接受。其次看不同事件的入流和出流差异，这反映降雨强度和管网响应的事件间变化。",
            "然而，后续消融结果显示 SWMM-assisted 没有优于 all-static。这并不意味着 SWMM 没有物理意义，而是说明“一个事件几个汇总数”太粗，无法告诉每个像元附近什么时候排水强、哪里可能出现瓶颈。要让 SWMM 更有效，下一步需要把节点水头、管段流量、节点溢流等时空变量映射回栅格，而不是只给模型事件级总量。"
        ],
    },
    {
        "id": "fig08",
        "file": "fig08_spatiotemporal_curves.png",
        "title": "测试事件时空过程曲线",
        "short": "本图从时间维度检查水量、误差和削减量变化，避免只看峰值空间图造成误判。",
        "detail": [
            "洪涝模型不是只预测一张最终地图，而是预测 6 小时内 72 个五分钟时刻的时空演化。图中的曲线用于回答：DrainLite 的修正是否只在某一个时刻偶然有效，还是能在整个事件过程中跟随排水削减变化。",
            "读时间序列图时，横轴是时间步或分钟，纵轴可能是平均水深、总水量、误差或削减量。若 ITZI surface-only 曲线长期高于 ITZI + sink，说明概念性排水入口持续减少了地表积水；若 DrainLite 曲线贴近 ITZI + sink，则说明残差模型捕捉到了这种随时间变化的削减。误差曲线在降雨峰后常常升高，因为地表汇流和排水响应在这一阶段最复杂。",
            "本图的结论是：DrainLite 对平均过程和总量变化有较好的跟随能力，但在峰值附近仍有偏差。这与模型设计一致。它用当前时刻的局部特征和邻域统计来预测削减量，没有真正求解二维浅水方程和一维管网方程之间的同步交换，因此对局部快速汇流、边界排放和节点壅水的时序响应仍然有限。"
        ],
    },
    {
        "id": "fig09",
        "file": "fig09_all_events_spatial_comparison.png",
        "title": "五个测试事件的全域峰值空间对比",
        "short": "本图把 event75、event76、event77、event78 和 event80 放在同一版面，检查结论是否跨事件稳定。",
        "detail": [
            "这张图是空间结果展示的主图。行通常对应不同测试事件，列通常对应不同模型或差值图，例如 MIKE reference、ITZI surface-only、ITZI + sink、DrainLite all-static、误差和预测削减量。它的作用类似原 LarNO 论文中的多模型洪水图：让读者同时看见空间分布、模型差异和事件间一致性。",
            "读每一行时，先看 MIKE reference 和 ITZI surface-only 的大体积水位置是否一致。如果低洼边界、道路洼地或主要汇水区域在两者中位置相近，说明地形和降雨驱动基本对上。再看 ITZI + sink 相对 surface-only 是否出现水深降低，这代表概念性排水入口的物理标签效果。最后看 DrainLite 是否在同一区域产生削减，而不是在无水或远离管网区域随意削水。",
            "读每一列时，可以比较不同事件的共性。若某列在五个事件中都出现类似改善，说明模型不是只记住某个事件。若某些强事件仍出现较大误差，说明削减规则受到降雨强度或汇流非线性的影响。整体来看，DrainLite 能在五个测试事件中稳定靠近 ITZI + sink 标签，尤其是大面积浅水区；但局部峰值和极端深水斑块仍然是短板。",
            "通俗地说，这张图告诉我们：这个轻量模型已经学会了“哪里像道路管网附近、哪里有水、哪里应该被排掉一点”，但还没有学会“每一处最深水坑在最强降雨后精确涨落多少”。这就是轻量残差模型和完整水动力耦合模型之间的差别。"
        ],
    },
    {
        "id": "fig10",
        "file": "fig10_peak_error_analysis.png",
        "title": "峰值误差与极端积水表现分析",
        "short": "本图专门检查平均误差改善是否掩盖了局部峰值问题。",
        "detail": [
            "平均绝对误差容易被大量浅水或干旱像元稀释，所以必须单独检查 peak error（峰值误差，表示预测最大水深与参考最大水深之间的差异）。在城市内涝预警中，局部最深点往往对应道路中断、地下空间进水或车辆危险，因此即便平均误差很小，峰值误差仍然不能忽略。",
            "读图时，如果某个面板展示峰值误差，越接近零越好；若展示 top 1% deep-cell MAE（最深 1% 像元平均绝对误差），它反映模型在最危险区域的表现，而不是全域平均表现；若展示峰值位置偏移，则表示最深点空间位置有没有跑偏。位置偏移为 0 米说明最深位置大体被找到，但水深数值仍可能偏差较大。",
            "本研究中 all-static 的平均峰值误差约 146.5 毫米，最深 1% 像元平均误差约 20.6 毫米。这个结果的意思是：DrainLite 对一般浅水削减很有效，但对单点最大水深仍不够强。原因可能包括概念性 sink 标签本身的局部不连续、树模型对极端样本的保守预测、以及没有完整二维-一维耦合反馈。论文写作时应把这一点作为限制，而不是回避。"
        ],
    },
    {
        "id": "fig11",
        "file": "fig11_drainage_physical_consistency.png",
        "title": "排水削减的物理一致性检查",
        "short": "本图检查 DrainLite 是否违反基本排水约束，例如预测水深高于 surface-only 或出现负水深。",
        "detail": [
            "这张图回答“模型虽然误差小，但有没有做出物理上奇怪的事”。对于当前 sink-only 目标，最基本的物理约束是：加入排水入口后，地表水深不应高于无管网地表结果；水深也不应小于零。因此我们检查 monotonic violation（单调性违背，指 h_drainlite 大于 h_surface 的像元）和 negative-depth violation（负水深违背，指预测水深小于零的像元）。",
            "读图时，若违背像元数为零，说明模型输出经过约束后满足基本物理边界。另一个重要面板是削减捕捉率，也就是 predicted reduction volume 与 true reduction volume 的比值。约 100% 表示总削减量接近标签；高于 100% 表示整体削得偏多；低于 100% 表示削得偏少。近管线削减占比则检查模型削减是否主要发生在管网附近。",
            "结果显示单调性和负水深违背均为 0，体积加权削减捕捉率约 107.1%，预测削减量约 78.0% 分布在距管线 60 米以内。这说明模型不是随意在全域抹平水深，而是大体沿着管网影响范围进行削减。不过 107.1% 也提示它略有过削倾向，尤其在弱事件或浅水区域需要谨慎。"
        ],
    },
    {
        "id": "fig12",
        "file": "fig12_event_severity_analysis.png",
        "title": "事件强度与模型误差关系",
        "short": "本图检查模型在强降雨和弱降雨事件上的稳定性。",
        "detail": [
            "不同降雨事件的强度、空间分布和持续时间不同，排水系统响应也会不同。事件强度分析的目的是避免一个平均指标掩盖“弱事件很好、强事件很差”或相反的情况。",
            "读图时，横轴可能是事件总降雨、峰值降雨或参考水量，纵轴是模型误差、削减误差或削减捕捉率。若点随事件强度增加而明显上升，说明模型在强事件下误差放大；若点分布平稳，说明泛化更稳定。",
            "本研究五个测试事件上的平均绝对误差范围约为 1.147 到 2.472 毫米，说明全域平均误差在事件之间相对稳定。但峰值误差和削减体积误差仍会随事件变化。论文中应强调：DrainLite 的优势主要体现在空间平均和管网削减分布层面，强事件极端峰值仍需要更多训练样本或真正耦合标签来增强。"
        ],
    },
    {
        "id": "fig13",
        "file": "fig13_mike_difference_maps.png",
        "title": "MIKE reference 外部参照差值图",
        "short": "本图展示 DrainLite 与 MIKE reference 的空间差异，用于外部合理性检查，而不是主训练目标评价。",
        "detail": [
            "这张图容易被误读，所以需要特别说明。MIKE reference 是原 LarNO 论文使用的参考来源，但 DrainLite 的监督标签是 ITZI + sink。两者并不代表同一个排水系统。MIKE 可能包含更完整的城市管网、不同边界条件和参数校准，而 ITZI + sink 是道路对齐概念性入口模型。",
            "读差值图时，正负色带表示某个模型相对 MIKE 偏高或偏低。若 DrainLite 更接近 ITZI + sink 但不一定更接近 MIKE，这不是简单的失败，而是说明“概念性排水入口标签”与“MIKE 参考体系”之间仍有结构差异。相反，如果某些区域 DrainLite 同时减少了对 MIKE 的偏差，可以视为外部 plausibility（合理性）增强。",
            "本图的结论应写得克制：DrainLite 可以作为外部参照下的对比对象，但不能宣称已经复现 MIKE 的完整一维/二维管网物理。要真正把 MIKE 作为目标，需要确认 MIKE 管网、边界、降雨和参数体系与概化管网的一致性，或者重新生成与本研究管网一致的高质量耦合 reference。"
        ],
    },
    {
        "id": "fig14",
        "file": "fig14_swmm_error_correlation.png",
        "title": "SWMM 指标与 DrainLite 误差相关性",
        "short": "本图探索一维管网事件级指标是否能解释 DrainLite 的误差变化。",
        "detail": [
            "相关性分析用于回答一个后续研究问题：如果 SWMM standalone 不直接改善像元级预测，它是否至少能解释哪些事件更难预测。图中的 Pearson correlation（皮尔逊相关系数，衡量两个变量线性同向或反向变化的程度，取值从 -1 到 1）用于描述 SWMM 指标与 DrainLite 误差之间的关系。",
            "读散点图时，每个点通常代表一个测试事件。横轴可能是 SWMM routing inflow 或 outfall volume，纵轴是 DrainLite 的平均绝对误差或削减误差。若点大体沿一条上升线排列，说明 SWMM 指标越大，误差也越大；若无明显趋势，说明事件级管网总量解释力有限。",
            "当前五个测试事件上，routing inflow 与 DrainLite 平均绝对误差的相关系数约 0.92，outfall volume 与削减误差的相关系数接近 1.00。但必须强调：样本数只有五个，这种高相关不能作为强统计结论。它更像一个线索，提示未来可把 SWMM 的时变节点水头、管段流量和溢流量转成空间特征，可能比事件总量更有用。"
        ],
    },
    {
        "id": "ext_swmm",
        "file": "extended_fig_itzi_swmm_prototype_summary.png",
        "title": "ITZI-PySWMM 双向耦合原型 17 事件汇总",
        "short": "本图展示 17 个有效事件的双向耦合原型审计结果，但仍不作为当前 DrainLite 主训练标签。",
        "detail": [
            "本图展示 17 个有效事件的 ITZI-PySWMM 原型统计。这里的 coupled prototype（耦合原型，指 ITZI 地表模型与 PySWMM 管网模型进行初步双向交换的试验性实现）已经能输出 72 帧水深，并记录 surface-to-pipe（地表进入管网的水量）和 pipe-to-surface（管网回到地表的水量）。与早期五事件探索相比，现在它已经完成 17 事件批量运行和统一 MIKE reference 对比，因此可以作为双向耦合路线的系统性原型证据。",
            "读图时应先看每个事件的入管量与回灌量，再看 coupled prototype 与 MIKE reference、ITZI surface-only、ITZI + sink 的误差。若 pipe-to-surface 大于 surface-to-pipe，说明原型中出现管网回灌、节点壅水或交换参数设置导致的地表补水；这与 sink-only DrainLite 的单调削减假设不同。因此它说明，完整耦合版本会比当前轻量残差模型更复杂，不能把“考虑管网”简单等同于“所有位置水深下降”。",
            "本次 17 事件汇总中，有 9 个事件表现为净回灌，即管网回到地表的累计水量大于地表进入管网的累计水量；另外 8 个事件表现为净排水。耦合原型相对 MIKE reference 的平均绝对误差约为 15.742 毫米。这个数值已经能说明原型具备可运行性和量级合理性，但也说明它离论文主参考标签仍有距离。报告中保留这张图的意义是把研究边界讲清楚：DrainLite 主模型学习的是概念性 sink 标签，而 ITZI-PySWMM 原型已经进一步展示了双向交换的可行性和复杂性。不过，该原型仍未经过真实管网参数校准、排放口边界校准和观测资料验证，因此还不能替代正式论文中的高置信度耦合 reference。"
        ],
    },
    {
        "id": "ext_events",
        "file": "extended_fig_event76_peak_maps.png",
        "title": "扩展图：event76 峰值空间对比",
        "short": "本扩展图补充单事件空间细节，便于检查 event76 是否与主图结论一致。",
        "detail": [
            "event76 的图读法与 event75 相同：先比较 MIKE、ITZI surface-only 和 ITZI + sink 的空间格局，再观察 DrainLite 是否在排水入口和管网附近产生合理削减。把单事件扩展图保留下来，是为了避免主图压缩后看不清局部细节。",
            "该图的价值在于提供可追溯性。如果某个汇总指标异常，研究者可以回到这类单事件图上检查是全域偏差、边界低洼区偏差、还是局部峰值偏差造成的。"
        ],
    },
    {
        "id": "ext_events77",
        "file": "extended_fig_event77_peak_maps.png",
        "title": "扩展图：event77 峰值空间对比",
        "short": "本扩展图用于观察较强事件中 DrainLite 的空间削减是否仍沿管网分布。",
        "detail": [
            "event77 的积水和削减总量较大，因此它能更清楚地暴露模型在强响应区域的行为。读图时重点看深水区的边界、管网附近浅水带和误差热点是否重合。",
            "如果误差主要集中在最深水斑块，而普通浅水区贴合较好，说明模型适合作为平均削减修正器，但仍需要更强的水动力表达处理极端点位。"
        ],
    },
    {
        "id": "ext_events78",
        "file": "extended_fig_event78_peak_maps.png",
        "title": "扩展图：event78 峰值空间对比",
        "short": "本扩展图补充 event78 的空间表现，用于跨事件稳定性检查。",
        "detail": [
            "event78 与 event77 一样属于测试集中削减量较大的事件。读图时应关注 DrainLite 是否出现过度削减，即某些区域比 ITZI + sink 更干。",
            "如果预测削减主要围绕管线和雨水口，而不是均匀铺满全域，则说明模型保留了空间先验；如果误差在低洼边界较明显，则可能提示边界开放性或地形洼地处理仍需改进。"
        ],
    },
    {
        "id": "ext_events80",
        "file": "extended_fig_event80_peak_maps.png",
        "title": "扩展图：event80 峰值空间对比",
        "short": "本扩展图补充 event80 的空间表现，是五个测试事件之一。",
        "detail": [
            "event80 用于验证模型在另一场未参与训练的事件上是否仍能保持相似行为。读图时同样按参考、基线、标签、预测、误差和削减量的顺序观察。",
            "若 event80 的整体误差与其他事件接近，就说明模型没有只对某一类事件有效；若个别热点偏差突出，则应在论文讨论中把它归入局部峰值和强非线性响应的限制。"
        ],
    },
    {
        "id": "ext_time",
        "file": "extended_fig_time_series.png",
        "title": "扩展图：完整时间序列对比",
        "short": "本扩展图提供比主文更细的时间过程检查。",
        "detail": [
            "完整时间序列图适合检查模型是否在降雨开始、峰值、退水三个阶段都保持合理。城市内涝常常不是降雨最大时立即达到最大水深，而是存在汇流滞后和排水滞后。",
            "如果 DrainLite 在退水阶段仍贴近 ITZI + sink，说明它学习到的削减不是只依赖瞬时降雨；如果峰后误差增大，说明缺少管网储水、回流和滞后释放机制。"
        ],
    },
    {
        "id": "fig15",
        "file": "fig15_paper_level_baseline.png",
        "title": "论文级轻量基线对比",
        "short": "本图补充统一削减率、距离衰减经验函数、线性模型、随机森林、极端随机树和梯度提升树等简单基线，证明 DrainLite 的选择不是偶然。",
        "detail": [
            "这张图对应原报告中的“待补充 5”。正式论文不能只给出一个 DrainLite 模型然后说它有效，还需要回答一个审稿人很自然会问的问题：如果用更简单的方法，比如全域统一削减一个比例，或者只按到排放口距离削减，效果会不会差不多？因此本图把多个轻量基线放到同一个评价框架下比较。",
            "读图时要注意，本图采用的是 stratified sample（分层抽样评价，即有意增加深水区、真实削减区、近管网区和普通背景区的样本比例），它不是全域平均像元指标。这样做的目的，是避免大量干区和浅水区把模型差异稀释掉。横轴是模型名称，纵轴分别展示样本平均绝对误差、深水峰值样本误差、样本峰值误差和 0.15 米阈值下的临界成功指数。",
            "从图中可以判断不同模型家族的能力边界。统一削减率和距离衰减经验函数如果表现较弱，说明管网削峰不能只靠一个全域比例或单一距离解释；线性模型如果误差较高，说明残差关系具有明显非线性；随机森林和极端随机树如果在抽样评价中表现较好，说明树模型确实适合这个轻量表格问题，也提示正式论文中可以把 DrainLite 更宽泛地表述为“管网残差学习框架”，而不是只绑定某一个具体回归器。"
        ],
    },
    {
        "id": "fig16",
        "file": "fig16_dense_peak_enhancement.png",
        "title": "峰值增强模型的全域密集检查",
        "short": "本图专门比较原 all-static HGB、peak-weighted HGB 和距离经验基线在完整 72 帧全域像元上的表现。",
        "detail": [
            "这张图对应原报告中的“待补充 4”。前期结果显示 DrainLite 的全域平均误差很低，但局部峰值水深误差仍然偏大。为了验证峰值误差能否通过轻量方法改善，本研究增加了 peak-weighted HGB（峰值加权直方图梯度提升树，即在训练时提高深水和高削减样本权重的版本）。",
            "读图时应把它和上一张抽样基线图区分开：这里是 dense full-domain evaluation（全域密集评价，即对每个测试事件的 72 个时刻和 200 x 280 全部格点计算预测），因此平均绝对误差会比分层抽样更低，因为干区和浅水区占比很高。若 peak-weighted HGB 的均方根误差下降但平均绝对误差略升，说明它在减少部分较大误差的同时，可能牺牲了浅水区的平均贴合。",
            "这张图的结论应写得细致：峰值加权不是无条件更好。它可以作为峰值增强方向的证据，但当前轻量版本仍不能完全解决局部极端峰值问题。下一步更有希望的路线，是把峰值加权与时序特征、邻域传播特征或小型卷积残差模块结合。"
        ],
    },
    {
        "id": "fig17",
        "file": "fig17_spatial_holdout_generalization.png",
        "title": "空间窗口 holdout 泛化验证",
        "short": "本图把 200 x 280 研究窗口分成四个空间子区，训练时排除一个子区，再在被排除子区评价。",
        "detail": [
            "这张图对应原报告中的“待补充 3”。原 DrainLite 实验按事件划分训练和测试，但空间窗口固定，因此还不能说明模型离开训练过的空间位置后是否仍然有效。空间 holdout（空间留出验证）就是把一个子窗口从训练样本中拿掉，只用其他区域训练，再到这个未见空间窗口上测试。",
            "读图时，左侧通常是四个空间窗口的位置；中间或右侧柱状图展示每个被留出窗口上的误差和相对 surface-only 的改善。若某个窗口误差明显更高，说明该区域的地形、管网密度、边界或汇流模式与训练区不同，模型泛化更困难。",
            "本研究中 southeast 窗口表现最弱，这一点很重要。它说明 DrainLite 并非在所有空间区域都同等可靠，也提示正式论文应增加对空间异质性的讨论。更进一步的改进可以包括按管网密度分层训练、加入区域编码、使用空间交叉验证选择模型，或构造更多不同窗口的数据。"
        ],
    },
    {
        "id": "fig18",
        "file": "fig18_pipe_density_strata.png",
        "title": "不同管网密度区域的误差分层",
        "short": "本图按管网密度把像元分成低、中、高三类，检查 DrainLite 在管网密集区是否更难预测。",
        "detail": [
            "这张图同样服务于空间泛化问题，但角度从“位置窗口”换成“管网结构密度”。管网密度高的区域通常意味着更多雨水口、更多管段、更多可能的排水路径和局部瓶颈；它既可能让排水削减更明显，也可能让模型更难预测。",
            "读图时，横轴是低管网密度、中等管网密度和高管网密度区域，纵轴展示平均误差、深水峰值误差或削减误差。若高密度区误差更大，不能简单说管网特征无效；更合理的解释是高密度区的排水作用更强、更局地化，轻量模型需要更细的节点水力状态和时序交换变量才能准确表达。",
            "这张图为论文讨论提供了很好的层次：DrainLite 已经能利用管网位置先验，但在管网最复杂的地方，静态栅格特征仍然不够。这个结果自然引出未来的 SWMM 时空特征和双向耦合 reference。"
        ],
    },
]


TABLES = [
    ("table1_dataset_model_configuration.csv", "表 1  数据集与模型配置", False),
    ("table2_ablation_metrics.csv", "表 2  消融实验指标", False),
    ("table3_mike_external_reference.csv", "表 3  MIKE reference 外部参照指标", False),
    ("table4_secondary_itzi_sink_metrics.csv", "表 4  ITZI + sink 主标签补充指标", False),
    ("table6_peak_error_analysis.csv", "表 5  峰值误差分析", False),
    ("table7_drainage_physical_consistency.csv", "表 6  排水物理一致性检查", False),
    ("table8_event_severity_analysis.csv", "表 7  事件强度分析", False),
    ("table9_swmm_error_correlation.csv", "表 8  SWMM 指标与误差相关性", False),
    ("table10_lightweight_runtime.csv", "表 9  轻量化运行资源统计", False),
    ("table11_paper_level_baseline_summary.csv", "表 10  论文级轻量基线汇总", False),
    ("table12_dense_peak_model_summary.csv", "表 11  峰值增强全域密集评价汇总", False),
    ("table13_spatial_holdout_summary.csv", "表 12  空间窗口 holdout 泛化汇总", False),
    ("table14_pipe_density_strata_summary.csv", "表 13  管网密度分层泛化汇总", False),
    ("extended_table1_event_inventory.csv", "附表 1  事件清单", True),
    ("extended_table2_feature_definitions.csv", "附表 2  管网与派生特征定义", True),
    ("extended_table3_coupled_prototype_summary.csv", "附表 3  ITZI-PySWMM 原型 17 事件统计", True),
    ("table5_time_series_metrics.csv", "附表 4  逐事件逐时刻时序指标明细", True),
]


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.3f}".rstrip("0").rstrip(".")
        if abs(value) >= 1:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def dataframe_to_html(path: Path) -> str:
    df = pd.read_csv(path)
    df = df.map(format_cell)
    return df.to_html(index=False, escape=True, classes="data-table")


def dataframe_to_markdown(path: Path) -> str:
    df = pd.read_csv(path)
    headers = list(df.columns)
    rows = [[format_cell(v) for v in row] for row in df.to_numpy()]
    def safe(x: str) -> str:
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(safe(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(safe(v) for v in row) + " |")
    return "\n".join(lines)


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def figure_html(fig: dict[str, object], idx: int) -> str:
    path = FIG_DIR / str(fig["file"])
    if not path.exists():
        return f"""
        <section class="figure-block missing">
          <h3>图 {idx}  {esc(fig['title'])}</h3>
          <p>待补充：未找到图像文件 {esc(path.name)}。</p>
        </section>
        """
    paragraphs = "\n".join(f"<p>{esc(p)}</p>" for p in fig["detail"])
    return f"""
    <section class="figure-block" id="{esc(fig['id'])}">
      <h3>图 {idx}  {esc(fig['title'])}</h3>
      <p class="figure-purpose">{esc(fig['short'])}</p>
      <img src="{image_data_uri(path)}" alt="{esc(fig['title'])}" />
      <div class="caption">
        {paragraphs}
      </div>
    </section>
    """


def figure_md(fig: dict[str, object], idx: int) -> str:
    path = FIG_DIR / str(fig["file"])
    rel = path.relative_to(PAPER_DIR).as_posix()
    parts = [f"### 图 {idx}  {fig['title']}", "", str(fig["short"]), ""]
    if path.exists():
        parts += [f"![{fig['title']}]({rel})", ""]
    else:
        parts += [f"待补充：未找到图像文件 `{path.name}`。", ""]
    for p in fig["detail"]:
        parts += [str(p), ""]
    return "\n".join(parts)


def table_html(filename: str, title: str, collapsible: bool) -> str:
    path = TABLE_DIR / filename
    if not path.exists():
        body = f"<p>待补充：未找到表格 {esc(filename)}。</p>"
    else:
        body = dataframe_to_html(path)
    note = TABLE_NOTES.get(filename, "本表直接来自本地计算输出，已嵌入 HTML，不依赖外部 CSV 文件。")
    content = f"""
    <section class="table-block">
      <h3>{esc(title)}</h3>
      <p>{esc(note)}</p>
      <div class="table-wrap">{body}</div>
    </section>
    """
    if collapsible:
        return f"""
        <details class="appendix-table">
          <summary>{esc(title)}：点击展开完整内嵌表格</summary>
          {content}
        </details>
        """
    return content


def table_md(filename: str, title: str, collapsible: bool) -> str:
    path = TABLE_DIR / filename
    parts = [f"### {title}", "", TABLE_NOTES.get(filename, "本表直接来自本地计算输出。"), ""]
    if path.exists():
        parts += [dataframe_to_markdown(path), ""]
    else:
        parts += [f"待补充：未找到表格 `{filename}`。", ""]
    return "\n".join(parts)


TABLE_NOTES = {
    "table1_dataset_model_configuration.csv": "本表交代研究对象的最小复现实验配置。重点是 20 米网格、72 个五分钟时刻、17 个有效事件和 12/5 训练测试划分。它决定了后续结果的适用范围，不能外推为真实 5 米 MIKE reference 验证。",
    "table2_ablation_metrics.csv": "本表是 DrainLite 是否成立的核心量化证据。比较不同特征组后可以看到 all_static 相对 surface_only 和 base 的误差下降，也可以看到 SWMM-assisted 并未进一步改善。",
    "table3_mike_external_reference.csv": "本表把各模型与 MIKE reference 对比。它是外部参照，不是主训练标签评价。读表时要注意：更接近 MIKE 不必然等于更接近 ITZI + sink，因为二者排水体系不同。",
    "table4_secondary_itzi_sink_metrics.csv": "本表补充相对于 ITZI + sink 标签的次要指标，用来检查平均误差以外的淹没范围和体积表现。",
    "table6_peak_error_analysis.csv": "本表专门列出峰值相关误差。它解释了为什么平均绝对误差已经很低，但局部最大水深仍需改进。",
    "table7_drainage_physical_consistency.csv": "本表检查排水削减是否满足非负水深、单调削减和近管网削减等基本物理一致性要求。",
    "table8_event_severity_analysis.csv": "本表检查不同测试事件强度下的误差变化，避免用一个平均数掩盖事件差异。",
    "table9_swmm_error_correlation.csv": "本表探索 SWMM standalone 事件级指标与 DrainLite 误差之间的相关性。由于测试事件只有五个，相关系数只能作为线索，不能作为强统计证明。",
    "table10_lightweight_runtime.csv": "本表展示本机轻量化运行成本。它说明 DrainLite 不依赖图形处理器，模型文件小、训练和推理时间低，但也明确 LarNO 和 MIKE 的参考运行时间来自原论文语境，不是同硬件直接对比。",
    "table11_paper_level_baseline_summary.csv": "本表补充正式论文需要的简单基线对比，包括统一削减率、距离衰减经验函数、线性模型、随机森林、极端随机树和梯度提升树。该表采用分层抽样评价，故数值不能直接与全域平均 MAE 混读。",
    "table12_dense_peak_model_summary.csv": "本表专门汇总全域密集峰值增强检查，比较 all-static HGB、peak-weighted HGB、距离经验基线与 surface-only。它用于判断峰值加权是否真正改善深水或峰值指标。",
    "table13_spatial_holdout_summary.csv": "本表汇总四个空间窗口 holdout 结果。训练样本排除被评价窗口，用于检验 DrainLite 对未见空间区域的泛化能力。",
    "table14_pipe_density_strata_summary.csv": "本表按管网密度分层统计误差，检查模型在低、中、高管网密度区域的表现差异。",
    "extended_table1_event_inventory.csv": "本附表列出 17 个有效事件及其训练/测试归属，是复现实验划分的依据。",
    "extended_table2_feature_definitions.csv": "本附表定义所有管网静态特征和派生邻域特征，便于后续论文写作时清楚说明每个输入变量的物理意义。",
    "extended_table3_coupled_prototype_summary.csv": "本附表列出 17 个有效事件的 ITZI-PySWMM 原型交换水量、误差和 MIKE reference 对比。它证明双向耦合流程已批量跑通，但仍需真实管网参数、边界和观测校准后才能升级为正式训练 reference。",
    "table5_time_series_metrics.csv": "本附表包含 5 个测试事件、72 个时间步的完整时序指标。正文不逐行讨论，但它保留了时间过程审计的全部数据。",
}


def load_key_numbers() -> dict[str, str]:
    numbers = {
        "all_static_mae": "待补充",
        "surface_mae": "待补充",
        "base_mae": "待补充",
        "improve_surface": "待补充",
        "improve_base": "待补充",
        "peak_error": "待补充",
        "capture_ratio": "待补充",
        "near_pipe": "待补充",
    }
    t2 = TABLE_DIR / "table2_ablation_metrics.csv"
    if t2.exists():
        df = pd.read_csv(t2)
        def row(model: str) -> pd.Series:
            return df.loc[df["model"] == model].iloc[0]
        all_static = row("all_static")
        surface = row("surface_only")
        base = row("base")
        numbers["all_static_mae"] = f"{all_static['mae_mm']:.3f} 毫米"
        numbers["surface_mae"] = f"{surface['mae_mm']:.3f} 毫米"
        numbers["base_mae"] = f"{base['mae_mm']:.3f} 毫米"
        numbers["improve_surface"] = f"{all_static['mae_improvement_vs_surface_pct']:.1f}%"
        numbers["improve_base"] = f"{all_static['mae_improvement_vs_base_pct']:.1f}%"
        numbers["peak_error"] = f"{all_static['peak_error_mm']:.1f} 毫米"
    t7 = TABLE_DIR / "table7_drainage_physical_consistency.csv"
    if t7.exists():
        df = pd.read_csv(t7)
        numbers["capture_ratio"] = f"{df['volume_weighted_capture_ratio_pct'].mean():.1f}%"
        numbers["near_pipe"] = f"{df['pred_reduction_near_pipe_share_pct'].mean():.1f}%"
    return numbers


def section(title: str, body: str, sec_id: str | None = None) -> str:
    attr = f' id="{esc(sec_id)}"' if sec_id else ""
    return f"<section{attr}><h2>{esc(title)}</h2>{body}</section>"


def p(text: str) -> str:
    return f"<p>{esc(text)}</p>"


def build_html() -> str:
    nums = load_key_numbers()
    figure_blocks = "\n".join(figure_html(fig, i + 1) for i, fig in enumerate(FIGURES))
    main_tables = "\n".join(table_html(f, t, c) for f, t, c in TABLES if not c)
    appendix_tables = "\n".join(table_html(f, t, c) for f, t, c in TABLES if c)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    toc = """
    <nav class="toc">
      <h2>目录</h2>
      <ol>
        <li><a href="#abstract">摘要</a></li>
        <li><a href="#terms">术语与符号说明</a></li>
        <li><a href="#background">研究背景与目的</a></li>
        <li><a href="#data-methods">数据与方法</a></li>
        <li><a href="#process">研究过程</a></li>
        <li><a href="#results">结果展示与图表解析</a></li>
        <li><a href="#discussion">分析与讨论</a></li>
        <li><a href="#conclusions">主要结论</a></li>
        <li><a href="#limitations">不足与展望</a></li>
        <li><a href="#appendix">附录：完整内嵌表格</a></li>
      </ol>
    </nav>
    """

    abstract = section("摘要", "\n".join([
        p("本报告围绕“如何在不重训完整 LarNO 主干模型的本机条件下，把城市排水管网效应纳入雨洪预报结果”这一问题展开。现有 LarNO 论文证明了潜在自回归神经算子可以在大尺度城市洪涝预报中实现高效推理，但本地硬件条件不支持全量神经算子训练，也缺少真实 5 米 MIKE reference 文件。因此，本研究转向一个轻量化、可解释、可复现的创新路径：以 ITZI surface-only 地表动力学结果为基底，以 ITZI + 概念性道路排水入口 sink 结果为监督标签，训练 DrainLite 排水残差修正模型。"),
        p(f"在 20 米网格、200 x 280 空间窗口、72 个五分钟时刻、17 个有效降雨事件的数据集上，本研究完成 12 个训练事件和 5 个测试事件的模型训练与评估。主模型 all_static 在 ITZI + sink 标签上的平均绝对误差为 {nums['all_static_mae']}，相比无管网 surface-only 基线的 {nums['surface_mae']} 降低 {nums['improve_surface']}，相比无管网特征的 base 残差模型的 {nums['base_mae']} 降低 {nums['improve_base']}。排水物理一致性检查显示，预测结果没有出现高于 surface-only 的单调性违背，也没有出现负水深；预测削减量约 {nums['near_pipe']} 分布在距管线 60 米以内，体积加权削减捕捉率约 {nums['capture_ratio']}。"),
        p(f"同时，本报告也明确指出局限：all_static 的平均峰值误差仍约 {nums['peak_error']}，说明轻量残差模型对全域平均削减和浅水区修正有效，但对局部极端水深峰值仍不足。SWMM standalone 的事件级指标在当前特征形式下没有带来额外精度提升，提示未来若要进一步利用 SWMM，应引入时空分布的节点水头、管段流量和溢流变量。ITZI-PySWMM 双向耦合原型已经扩展到 17 个有效事件，但仍缺少真实管网参数、排放口边界和观测资料校准，因此尚不能作为完整双向耦合 reference。"),
    ]), "abstract")

    terms = section("术语与符号说明", """
    <div class="term-grid">
      <div><strong>LarNO（Latent Autoregressive Neural Operator，潜在自回归神经算子）</strong><p>它把城市洪涝过程看作从降雨、地形等输入函数到水深输出函数的连续算子映射。物理上，它试图学习二维地表水动力系统的时空响应；理论上，它继承神经算子的函数空间映射思想，并通过 latent autoregression（潜在空间自回归，即在隐藏特征而不是单一水深变量中传递历史状态）缓解长时序误差累积。本研究不重训 LarNO，而是沿用其论文的数据组织和参考思想，提出本机可运行的排水残差层。</p></div>
      <div><strong>DrainLite（轻量排水残差修正模型）</strong><p>这是本研究提出的轻量模型名称。它的物理意义是：在已有地表洪涝结果上，学习排水管网能移走多少水。方程上，它学习 r=max(h_surface-h_sink,0)，再用 h_drainlite=max(h_surface-r_hat,0) 得到修正水深。引入它的原因是本机 GPU 无法支撑完整 LarNO-D 训练，但仍希望用管网特征完成可统计、可消融、可解释的创新。</p></div>
      <div><strong>ITZI（本研究中的地表二维雨洪动力学求解器，英文缩写全称待补充）</strong><p>物理上用于模拟降雨在数字高程模型上的地表汇流、积水和退水过程。它不是简单洼地填充，而是动态地表过程求解。本文把 ITZI surface-only 作为无管网基底，把 ITZI + sink 作为概念性排水入口标签。缩写的官方全称需要在正式论文中从 ITZI 文档补充确认。</p></div>
      <div><strong>MIKE Plus（商业城市水动力与排水系统建模软件）</strong><p>原 LarNO 论文中使用 MIKE reference 作为数值参考结果。它可表示一维管网和二维地表的耦合。本文中 MIKE 只作为外部参照，不作为 DrainLite 主训练标签，因为本研究的概化管网与 MIKE 原始管网体系不完全一致。</p></div>
      <div><strong>SWMM（Storm Water Management Model，雨洪管理模型）</strong><p>SWMM 用于一维排水管网水动力计算。dynamic wave（动力波）求解完整一维 Saint-Venant 方程中的惯性、压力、重力和摩阻效应，比简化运动波更适合有回水和满流的管网。本文的 SWMM standalone 是独立管网分支，只提供事件级指标，不等同于完整地表-管网双向耦合。</p></div>
      <div><strong>DEM（Digital Elevation Model，数字高程模型）</strong><p>DEM 是每个栅格的地面高程。它在水动力方程中决定坡度和重力势能差，控制水向低处汇流。本文所有排水特征必须与 DEM 对齐，否则模型会在错误位置预测排水削减。</p></div>
      <div><strong>sink（源汇项中的汇，即排水入口移除项）</strong><p>在水量连续方程中，source/sink term（源汇项）表示外部向系统加入或移除水。降雨是源项，排水入口是汇项。ITZI + sink 标签通过概念性道路雨水口从地表移走水量，用于构造 DrainLite 的监督残差。</p></div>
      <div><strong>MAE（Mean Absolute Error，平均绝对误差）</strong><p>它是所有像元和时刻上预测值与参考值差的绝对值平均。物理上可理解为平均每个格点水深差多少。它易读但会被大量浅水或干旱像元稀释，因此本文同时报告峰值误差和深水区指标。</p></div>
      <div><strong>RMSE（Root Mean Square Error，均方根误差）</strong><p>它先平方误差再平均再开方，对大误差更敏感。若 RMSE 明显大于 MAE，说明少数区域存在较大的局部误差。</p></div>
      <div><strong>CSI（Critical Success Index，临界成功指数）</strong><p>它用于判断淹没范围分类是否准确，公式为 TP/(TP+FP+FN)。TP 是正确预测为淹没的格点，FP 是误报淹没，FN 是漏报淹没。本文使用 0.03 米和 0.15 米两个阈值，分别关注浅水内涝和较显著积水。</p></div>
      <div><strong>Base64（用于把图片编码进 HTML 的文本格式）</strong><p>Base64 把二进制图片转成文本字符串，使 HTML 文件不再依赖外部图片路径。本文所有 PNG 图都以内嵌 Base64 形式写入报告，因此 report.html 可以单文件迁移。</p></div>
    </div>
    """, "terms")

    background = section("研究背景与目的", "\n".join([
        p("原 LarNO 论文的写法特点是先说明城市尺度高分辨率洪涝预报的计算瓶颈，再把问题抽象为连续算子学习，随后用 MIKE reference、时空水深图、误差曲线、临界成功指数、消融实验和效率对比支撑结论。本研究若要形成新的论文，也需要沿着类似逻辑组织证据：先指出管网效应对城市内涝预报的重要性，再说明完整一维/二维耦合和全量神经算子训练在本机条件下不可行，最后提出一个更轻量但仍可验证的替代方案。"),
        p("本研究的目标不是声称已经完成真实城市排水系统的全量重建，也不是把概化管网直接等同于规划管网。更准确的定位是：构建道路对齐概化管网特征，利用 ITZI + sink 生成受控排水削峰标签，训练一个不依赖 GPU 的残差学习器，使已有地表洪涝结果具备管网感知的削峰修正能力。这个创新点小而清楚，适合在当前本机条件下完成系统化验证。"),
        p("从论文写作角度看，最重要的边界是三分法：ITZI + sink 是主监督标签；MIKE reference 是外部参照；SWMM standalone 是管网物理指标。把这三者分清，论文就不会把合成标签、商业参考和独立管网计算混为一谈。"),
    ]), "background")

    data_methods = section("数据与方法", "\n".join([
        p("数据集命名为 region1_20m_drainage_v1，保持 LarNO benchmark 的目录风格。每个事件包含 rainfall.npy、h_mike_ref.npy、h_itzi_surface.npy、h_itzi_sink.npy 或 h.npy 等结果；地理数据包含 dem.npy 和八个管网栅格特征。事件总数为 17，其中 event79 因本地文件损坏被排除。训练事件 12 个，测试事件 5 个。"),
        p("空间分辨率为 20 米，空间窗口为 200 x 280 个格点，单个时刻共有 56,000 个像元；时间分辨率为 5 分钟，共 72 帧，对应 6 小时事件过程。当前本地缺少真实 region1_5m MIKE reference 文件，因此报告不把 5 米结果作为正式验证，只把 20 米结果作为主实验。"),
        p("DrainLite 的训练目标是毫米单位削减量：target_reduction_mm=1000*max(h_itzi_surface-h_itzi_sink,0)。采用毫米单位是为了让回归目标数值更适合树模型训练；最终输出再除以 1000 转回米。模型预测后执行 h_drainlite=max(h_itzi_surface-pred_reduction_mm/1000,0)，并裁剪保证 h_drainlite 不高于 h_itzi_surface。"),
        p("模型使用 HistGradientBoostingRegressor（直方图梯度提升回归树，一种在 CPU 上高效训练的非线性表格模型）。它不是深度神经网络，但能学习地形、水深、降雨和管网特征之间的非线性组合，并且可以做消融和特征重要性分析。"),
    ]), "data-methods")

    process = section("研究过程", """
    <ol class="process-list">
      <li><strong>复现与数据理解。</strong>先梳理 LarNO 论文、MIKE reference、20 米数据、5 米数据缺失情况和已有结果文件，明确当前正式评价只能基于本地可用 20 米数据。</li>
      <li><strong>地形、路网和管网对齐。</strong>针对早期出现的上下翻转、整体偏移和路网微调问题，最终采用当前已校正版本，并以 DEM + 管网叠加图进行空间审计。</li>
      <li><strong>物理标签分层。</strong>保留 ITZI surface-only 作为无管网基线，使用 ITZI + conceptual inlet sink 作为主训练标签，使用 MIKE reference 作为外部参照，使用 SWMM standalone 指标作为管网事件级解释变量。</li>
      <li><strong>轻量模型训练。</strong>在 12 个训练事件上采样像元训练 DrainLite，并在 5 个未见测试事件上输出完整 72 x 200 x 280 预测数组。</li>
      <li><strong>消融与解释。</strong>比较 base、mask-only、hydraulic-only、all-static 和 SWMM-assisted，判断管网位置、管网水力参数和 SWMM 事件级指标各自贡献。</li>
      <li><strong>双向耦合原型探索。</strong>17 个有效事件的 ITZI-PySWMM 原型已批量跑通，并完成统一统计和 MIKE reference 对比；但它仍作为原型路线，不写成已校准的完整耦合 reference。</li>
      <li><strong>论文与报告整理。</strong>将图、表、公式、边界声明和结果解释整理为自包含报告，并同步保留 Markdown 与 PDF 输出。</li>
    </ol>
    """, "process")

    results = section("结果展示与图表解析", figure_blocks + "\n" + main_tables, "results")

    discussion = section("分析与讨论", "\n".join([
        p("第一，DrainLite 的主要成功不是“超过 MIKE”，而是在受控标签上证明道路对齐管网特征可以显著提升地表模型向带排水入口模型的映射。all_static 相对 surface-only 的误差下降说明，排水削峰不是简单随机噪声；相对 base 的改进说明，管网特征提供了地形、降雨和已有水深之外的信息。"),
        p("第二，mask-only 接近 all-static 是一个值得写进论文的发现。它说明在当前概化管网条件下，位置先验比参数精细度更关键。换句话说，模型首先需要知道“哪里可能排水”，其次才是“管径坡度具体是多少”。这与当前管网是概化生成而非真实资产有关，参数场可能噪声较大，因此没有表现出强优势。"),
        p("第三，SWMM-assisted 没有进一步提升，不应被简单写成失败。更合理的解释是：事件级 SWMM 总量过于粗糙，无法提供像元级排水差异。这个结果反而指出了后续升级路径，即将 SWMM 的节点水头、节点溢流、管段流量和局部可用容量转成时空栅格特征。"),
        p("第四，峰值误差较大说明轻量残差模型仍不是完整水动力耦合模型。真实峰值受局部地形、边界、汇流路径、管网满流、回灌和时间滞后共同控制。DrainLite 当前以逐像元残差和局部邻域统计为主，缺少显式水量传播和一维/二维交换方程，因此适合作为轻量修正层，而不是最终物理 reference。"),
        p("第五，从论文选题角度，当前最稳妥的题目应围绕 low-compute drainage-aware residual correction，而不是 full drainage-coupled neural operator。后者现在已有 17 事件双向耦合原型作为基础，但若要成为论文主标签，还需要真实管网参数校准、严格水量守恒审计、排放口边界校准、与 MIKE 参考的一致性解释和更多空间泛化实验。"),
    ]), "discussion")

    conclusions = section("主要结论", """
    <ol class="conclusion-list">
      <li>本研究完成了一个本机可运行的 LarNO-DrainLite 轻量排水残差修正流程，避免了全量 LarNO-D 训练对高端图形处理器的依赖。</li>
      <li>在五个未见测试事件上，all_static 模型相对 ITZI + sink 标签的平均绝对误差为约 1.946 毫米，相比 surface-only 基线下降约 66.7%，相比无管网特征残差模型下降约 18.2%。</li>
      <li>消融实验表明，管网位置特征是最主要贡献来源；当前概化管网的水力参数和 SWMM 事件级汇总指标没有带来同等幅度的增益。</li>
      <li>物理一致性检查显示，DrainLite 输出满足非负水深和不高于 surface-only 的单调削减约束，且削减主要分布在管网附近。</li>
      <li>模型对全域平均削减有效，但局部峰值误差仍显著，说明它不能替代完整的一维/二维地表-管网双向耦合模型。</li>
      <li>17 个有效事件的 ITZI-PySWMM 原型说明进一步构造双向耦合标签具有可行性，但尚需补充真实参数、排放口边界、守恒审计和观测校准。</li>
    </ol>
    """, "conclusions")

    limitations = section("不足与展望", "\n".join([
        p("待补充 1：真实管网资料。当前管网来自道路对齐概化提取，缺少官方管径、埋深、节点井底、排放口边界和运行调度信息。若要形成更强论文，需要至少补充一组真实或半真实管网对照，或明确证明概化管网作为先验的适用范围。"),
        p("待补充 2：双向耦合 reference。当前 ITZI-PySWMM 已完成 17 个有效事件的原型批处理和 MIKE reference 统一比较，但仍不能替代 ITZI + sink 标签。下一步需要补充真实管网参数、排放口边界条件、质量守恒审计、节点交换机制敏感性检查，并解释哪些事件表现为净排水、哪些事件表现为满管回灌。"),
        p("待补充 3：空间泛化。当前训练测试按事件划分，空间窗口固定。若要接近 LarNO 论文的完整度，应增加不同窗口、不同区域或不同管网密度的泛化验证。"),
        p("待补充 4：峰值预报增强。局部峰值误差仍大，后续可尝试峰值加权损失、深水区过采样、时序特征、邻域传播特征或小型卷积残差模块。"),
        p("待补充 5：论文级对比。正式投稿还需要补充与更多简单基线的比较，例如统一削减率、距离管网经验函数、随机森林、极端梯度提升模型或小型神经网络，以证明 DrainLite 的选择不是偶然。"),
    ]), "limitations")

    appendix = section("附录：完整内嵌表格", appendix_tables, "appendix")

    css = """
    :root {
      --ink: #18212f;
      --muted: #5e6878;
      --line: #d9e0ea;
      --soft: #f5f7fb;
      --accent: #1f6f8b;
      --accent-2: #784f9f;
      --good: #247a4d;
      --warn: #b16a00;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: #ffffff;
      font-family: "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", Arial, sans-serif;
      line-height: 1.72;
      font-size: 16px;
    }
    .page {
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 30px 80px;
    }
    .cover {
      min-height: 88vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      border-bottom: 4px solid var(--ink);
      padding: 70px 0 50px;
    }
    .eyebrow {
      color: var(--accent);
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
      margin-bottom: 24px;
    }
    h1 {
      font-size: 42px;
      line-height: 1.22;
      margin: 0 0 24px;
      max-width: 980px;
    }
    .subtitle {
      font-size: 20px;
      color: var(--muted);
      max-width: 920px;
      margin: 0 0 36px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      max-width: 900px;
    }
    .meta div, .callout, .toc, .term-grid div {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }
    .toc {
      margin: 34px 0;
    }
    .toc h2 { margin-top: 0; }
    .toc a { color: var(--accent); text-decoration: none; }
    section {
      margin: 42px 0;
    }
    h2 {
      font-size: 28px;
      border-left: 6px solid var(--accent);
      padding-left: 14px;
      margin: 0 0 18px;
    }
    h3 {
      font-size: 21px;
      margin: 26px 0 10px;
    }
    p { margin: 0 0 14px; }
    .term-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }
    .term-grid p { margin: 8px 0 0; color: #2b3544; }
    .process-list li, .conclusion-list li {
      margin-bottom: 10px;
    }
    .figure-block {
      border-top: 1px solid var(--line);
      padding-top: 24px;
      margin-top: 36px;
    }
    .figure-purpose {
      color: var(--accent-2);
      font-weight: 700;
    }
    .figure-block img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      margin: 14px 0 18px;
    }
    .caption {
      border-left: 4px solid var(--line);
      padding-left: 16px;
      color: #283445;
    }
    .table-block {
      margin: 28px 0;
      border-top: 1px solid var(--line);
      padding-top: 18px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table.data-table {
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      line-height: 1.45;
    }
    .data-table th {
      background: #eaf1f7;
      color: #142033;
      text-align: left;
      position: sticky;
      top: 0;
    }
    .data-table th, .data-table td {
      border-bottom: 1px solid #e6ebf2;
      padding: 8px 10px;
      vertical-align: top;
      white-space: nowrap;
    }
    .data-table tr:nth-child(even) td { background: #fafbfd; }
    details.appendix-table {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 16px;
      margin: 14px 0;
      background: #fbfcfe;
    }
    details summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
    }
    .footer {
      margin-top: 60px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 14px;
    }
    @media print {
      body { font-size: 13px; }
      .page { max-width: none; padding: 0 18mm; }
      .cover { min-height: 0; page-break-after: always; }
      section, .figure-block, .table-block { break-inside: avoid; }
      details.appendix-table { break-inside: auto; }
      a { color: inherit; }
    }
    """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DrainLite 城市排水管网感知轻量洪涝预报残差修正研究报告</title>
  <style>{css}</style>
</head>
<body>
  <main class="page">
    <header class="cover">
      <div class="eyebrow">Standalone Research Report</div>
      <h1>DrainLite 城市排水管网感知轻量洪涝预报残差修正研究报告</h1>
      <p class="subtitle">基于 LarNO 深圳城市洪涝数据、ITZI 动力学结果、道路对齐概化管网、SWMM standalone 指标与本机可运行轻量模型的论文式整理</p>
      <div class="meta">
        <div><strong>研究定位</strong><br />低算力条件下的管网效应注入与残差学习</div>
        <div><strong>主监督标签</strong><br />ITZI + 概念性道路排水入口 sink</div>
        <div><strong>外部参照</strong><br />MIKE reference，非主训练目标</div>
        <div><strong>生成时间</strong><br />{esc(generated_at)}</div>
      </div>
    </header>
    {toc}
    {abstract}
    {terms}
    {background}
    {data_methods}
    {process}
    {results}
    {discussion}
    {conclusions}
    {limitations}
    {appendix}
    <footer class="footer">
      本 HTML 为完全自包含单文件报告：图片已转换为 Base64 内嵌，表格已直接写入 HTML，样式写在 head/style 内部。Markdown 与 PDF 文件由同一脚本同步生成。
    </footer>
  </main>
</body>
</html>
"""


def build_markdown() -> str:
    nums = load_key_numbers()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = [
        "# DrainLite 城市排水管网感知轻量洪涝预报残差修正研究报告",
        "",
        f"生成时间：{generated_at}",
        "",
        "## 摘要",
        "",
        f"本报告围绕低算力条件下如何把城市排水管网效应纳入雨洪预报结果展开。DrainLite 以 ITZI surface-only 地表动力学结果为基底，以 ITZI + 概念性道路排水入口 sink 结果为监督标签，学习非负排水削减量。在 17 个有效事件、20 米网格、72 个五分钟时刻的数据集上，all_static 模型在 5 个测试事件上的平均绝对误差为 {nums['all_static_mae']}，相比 surface-only 下降 {nums['improve_surface']}，相比 base 下降 {nums['improve_base']}。",
        "",
        "## 术语说明",
        "",
        "LarNO 是 Latent Autoregressive Neural Operator，即潜在自回归神经算子。DrainLite 是本研究提出的轻量排水残差修正模型。ITZI 在本文中指本地使用的地表二维雨洪动力学求解器，官方缩写全称待补充。MIKE Plus 是原 LarNO 论文中的外部水动力 reference。SWMM 是 Storm Water Management Model，用于独立一维管网动力波计算。DEM 是 Digital Elevation Model，即数字高程模型。sink 是水量方程中的汇项，表示概念性排水入口移除地表水。",
        "",
        "## 研究背景与目的",
        "",
        "原 LarNO 论文强调大尺度高分辨率城市洪涝预报的神经算子路线。本研究在本机 GPU 不适合全量训练的条件下，转向可复现、可消融、可解释的轻量创新：不重训完整 LarNO 主干，而是在已有地表洪涝结果之后叠加管网感知排水残差修正。",
        "",
        "## 数据与方法",
        "",
        "数据集为 region1_20m_drainage_v1，包含 17 个有效事件，其中 12 个用于训练，5 个用于测试。空间窗口为 200 x 280，分辨率为 20 米，时间长度为 72 步。训练目标为 target_reduction_mm=1000*max(h_itzi_surface-h_itzi_sink,0)，预测后用 h_drainlite=max(h_itzi_surface-pred_reduction_mm/1000,0) 得到修正水深。",
        "",
        "## 研究过程",
        "",
        "1. 梳理 LarNO 论文与本地数据可用性。\n2. 校正 DEM、路网和管网相对位置。\n3. 分离 ITZI surface-only、ITZI + sink、MIKE reference 和 SWMM standalone 的角色。\n4. 训练 DrainLite 并完成消融实验。\n5. 输出空间图、时序图、统计表、物理一致性检查和原型耦合证据。\n6. 生成 HTML、Markdown 和 PDF 报告。",
        "",
        "## 图表解析",
        "",
    ]
    for i, fig in enumerate(FIGURES, 1):
        parts.append(figure_md(fig, i))
    parts += ["## 表格", ""]
    for filename, title, collapsible in TABLES:
        parts.append(table_md(filename, title, collapsible))
    parts += [
        "## 主要结论",
        "",
        "- DrainLite 在受控 ITZI + sink 标签上显著降低平均误差，说明道路对齐概化管网特征能够提供排水削峰信息。",
        "- 管网位置特征贡献最大，事件级 SWMM standalone 汇总指标在当前形式下没有进一步提高像元级精度。",
        "- 模型满足非负水深和不高于 surface-only 的单调约束，但局部峰值误差仍需要通过更强时空模型或完整耦合标签改善。",
        "- 当前 17 事件 ITZI-PySWMM 结果是批量原型证据，不应写成已经校准验证后的完整双向耦合 reference。",
        "",
        "## 不足与展望",
        "",
        "待补充真实管网资料、双向耦合原型的参数/边界/守恒校准、空间泛化测试、峰值加权训练和更多轻量基线比较。",
    ]
    return "\n".join(parts)


def check_self_contained(html_text: str) -> None:
    if "<!DOCTYPE html>" not in html_text[:100]:
        raise RuntimeError("HTML missing <!DOCTYPE html> declaration.")
    if "<style>" not in html_text or "</style>" not in html_text:
        raise RuntimeError("HTML missing internal CSS style block.")
    if re.search(r'<img[^>]+src=["\'](?!data:image/png;base64,)', html_text, re.I):
        raise RuntimeError("Found non-base64 image source in HTML.")
    if re.search(r'https?://', html_text, re.I):
        raise RuntimeError("Found network URL in HTML.")


def find_browser() -> str | None:
    candidates = [
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("chromium"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def render_pdf() -> bool:
    browser = find_browser()
    if not browser:
        return False
    if OUT_PDF.exists():
        OUT_PDF.unlink()
    user_data = PAPER_DIR / ".browser_pdf_profile"
    user_data.mkdir(exist_ok=True)
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        f"--user-data-dir={user_data}",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.resolve().as_uri(),
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=False, timeout=180)
    if result.returncode != 0:
        print(result.stdout.decode("utf-8", errors="replace") if result.stdout else "")
        print(result.stderr.decode("utf-8", errors="replace") if result.stderr else "")
        return False
    return OUT_PDF.exists() and OUT_PDF.stat().st_size > 10_000


def main() -> None:
    html_text = build_html()
    check_self_contained(html_text)
    OUT_HTML.write_text(html_text, encoding="utf-8")
    OUT_MD.write_text(build_markdown(), encoding="utf-8")
    pdf_ok = render_pdf()
    shutil.copy2(OUT_HTML, ROOT_HTML)
    shutil.copy2(OUT_MD, ROOT_MD)
    if pdf_ok:
        shutil.copy2(OUT_PDF, ROOT_PDF)
    print(f"HTML: {OUT_HTML} ({OUT_HTML.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Markdown: {OUT_MD} ({OUT_MD.stat().st_size / 1024 / 1024:.2f} MB)")
    if pdf_ok:
        print(f"PDF: {OUT_PDF} ({OUT_PDF.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"Root copies: {ROOT_HTML}, {ROOT_MD}, {ROOT_PDF}")
    else:
        print("PDF: generation failed or no browser found")


if __name__ == "__main__":
    main()
