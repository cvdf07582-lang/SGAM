[2305.00194v6.pdf](https://leedong25.yuque.com/attachments/yuque/0/2025/pdf/60748161/1764429673293-0eb663c0-6b7a-43b6-8362-633e7ff69d67.pdf)

github[https://github.com/Easonyesheng/SGAM](https://github.com/Easonyesheng/SGAM)

airvix[https://arxiv.org/abs/2305.00194](https://arxiv.org/abs/2305.00194)

### 论文介绍

特征匹配是计算机视觉中的一项基本任务，是多种视觉应用的基础，如同时定位和映射（SLAM），运动结构（SfM）以及图像对齐。尽管这项任务被广泛研究，准确确定单个三维点在两个不同视角下的投影仍是挑战。这些挑战源于**匹配噪声**，如潜在的极端视角、光线变化、重复图案和运动模糊，这些都导致匹配精度有限。

当前特征匹配方法分为**稀疏法、半稠密法和稠密法**。尽管具体技术不同，这些方法都遵循一个共识：**采用 coarse-to-fine 的分层匹配策略**，即先建立中间的搜索空间（intermediate search space），然后在该空间内完成点级匹配。

* **Sparse methods（稀疏方法）**：先在图像中检测关键点集合，再基于检测的点及其描述子建立对应关系；论文指出即使使用深度 CNN 检测，关键点检测在存在匹配噪声（如极端视角、光照、模糊等）时仍可能不准确或失败，从而导致方法在极端情况下失效。
* **Semi-dense / Dense methods（半稠密与稠密方法）**：通过密集的 patch-wise 特征比较来获得更细粒度的匹配（例如利用 4D CNN、Transformer 等架构），半稠密方法常先计算密集特征再筛选为稀疏 patch 以精炼点匹配；但论文指出这种 **patch 匹配依赖于大规模的 dense 特征比较，会引入对非重叠/无关 patch 的错误比较（冗余计算与噪声）并限制输入分辨率**，从而影响整体精度。

论文还提到已有的“两阶段 / co-visible area”方法试图估计重叠区域作为中间搜索空间，但这些方法仍高度依赖特征比较，因而受匹配噪声影响。PATS 等方法通过分割/裁切补救分辨率问题，但仍存在不必要的冗余比较。

**核心瓶颈：中间搜索空间（intermediate search space）不可靠**

论文讨论了将语义引入匹配的相关工作，指出尽管语义信息相对稳健（对光照、视角等噪声更不敏感），**已有方法通常只是将语义用来增强 patch 或 keypoint 的特征表示，但这些方法并未改造或重新定义中间搜索空间本身。**

论文进一步说明：语义分割的边界并不精确（存在边界噪声、标签分割误差），而细粒度的匹配通常聚集在这些边界处，因此**把语义直接用于细粒度搜索空间或仅作为特征增强会受到语义边界噪声和 semantic drift 的负面影响。**同时，多实例（同类物体多次出现）会产生语义歧义，使得单纯语义匹配可能有多个候选或错误候选，需要额外机制来去解歧。

为了解决上述瓶颈，论文提出：**将语义作为中间搜索空间的设计基石——即构建一个“semantic-friendly” 的中间搜索空间，称为 Semantic Area Matches（语义区域匹配）**。该搜索空间由“具有显著语义信息的匹配图像区域”组成，能够基于语义的不变性较为鲁棒地建立，从而减少冗余计算并允许在这些区域中使用更高分辨率输入做精细点匹配。

尽管 SAM 能有效检测并匹配图像间的大部分语义区域，但语义学中固有的抽象特性忽略了局部细节。这可能导致匹配时的语义模糊，尤其是在图像中存在不同实例时。因此，**SAM 可以识别出无法确定匹配的可疑区域（doubtful areas）。**此外，语义歧义还可能导致 SAM 中的区域匹配错误，从而对特征匹配产生不利影响。

**所以必须引入新的约束：GAM（Geometric Area Matching）几何一致性解决语义歧义。** GAM 包含三部分：

1.  **GP（Predictor）**：在多个可疑匹配中找出最可能正确的一组。
2.  **GR（Rejector）**：用几何一致性丢掉错误区域。
3.  **GMC（Global Match Collection）**：当语义区域太少时，从全图补充一些点匹配。

**结论：SGAM = SAM + GAM**

| 部分 | 作用 |
| --- | --- |
| **SAM** | 用语义进行区域匹配（对象 + 交叉区域） |
| **GP** | 歧义区域中挑出正确区域对 |
| **GR** | 用几何一致性过滤错误区域 |
| **GMC** | 区域太少时补充全图点匹配 |

---

**论文贡献如下：**

1.  引入一种语义友好的中间搜索空间用于特征匹配，称为语义区域匹配，并配套一个名为 A2PM 的匹配框架。该框架涉及先建立图像间的语义区域匹配，然后在这些区域图像对内进行点匹配，最终提高匹配准确性。
2.  为实现 A2PM 框架，我们提出了 SGAM 方法，该方法由两个组成部分组成：SAM，负责根据语义识别假定的面积匹配，以及 GAM，通过确保几何形状一致性来获得精确的面积和点匹配。
3.  利用基于 LLM 的语义切割方法，SGAM 在稀疏、半稠密和密集匹配方法的匹配准确性上持续提升，从而在各种室内外数据集上取得了令人印象深刻的姿态估计表现。

---

### 相关工作

#### 1. 稀疏匹配 (Sparse Matching)

稀疏方法主要依赖**关键点检测 + 描述子匹配**。代表方法包括 SuperPoint [7]、LF-Net [8]、D2-Net [9]、R2D2 [10]、DISK [11]、ASLFeat [12]。这些方法首先通过深度网络检测关键点，然后提取局部描述子，再通过最近邻搜索建立匹配，并结合 OANet [13]、NG-RANSAC [14] 等学习式方法进行外点剔除。

* **原文指出的缺点**：稀疏方法的 coarse 搜索空间由“关键点集合”决定，在低纹理、重复纹理、极端视角变化情况下关键点检测不稳定，从而导致 coarse 空间不可靠，最终影响点匹配性能。
* **本文的改进方向**：本文提出使用**语义区域匹配（Semantic Area Matches, SAM）**来替代关键点集合作为 coarse 搜索空间，使 coarse 区域具有语义一致性，从而提高匹配的稳定性，并结合 GAM 进一步增强几何一致性。

#### 2. 半稠密匹配与稠密匹配 (Semi-dense and Dense Matching)

稠密方法如早期 4D CNN 方法 [15]、DKM [16]、COTR [17]，会对大量像素或 patch 进行特征比较。半稠密方法如 LoFTR [9] 以及其后续变体（QuadTree Attention [19]、ASpanFormer [20]）采用 coarse-to-fine 结构，通过在稠密 coarse 空间中建立粗匹配，再细化得到稀疏点匹配。

* **原文指出的缺点**：dense/semi-dense 方法需要在宽广范围的特征图上计算和比较，造成大量冗余计算，尤其在无关区域中；同时其 coarse-level 分辨率受到特征下采样限制，使后续匹配精度受限。
* **本文的改进方向**：通过使用语义区域构建 coarse 搜索空间，只比较语义相关区域，避免无关区域的冗余代价，并允许在裁剪出的语义区域上使用更高分辨率输入，从而提高匹配精度。

#### 3. 粗到精匹配 (Coarse-to-Fine Matching)

许多现代方法遵循 coarse-to-fine 框架：semi-dense 方法使用 coarse patch matching（如 LoFTR [9]），dense warp 方法在 coarse feature map 上计算全局对应（COTR [17]），另一些方法如 PATS [21]、OETR/Overlap Estimation [22,23] 和其他两阶段区域方法 [24] 则先预测区域/重叠区域再执行精细点匹配。

　　许多现代方法遵循 coarse-to-fine 框架：semi-dense 方法使用 coarse patch matching（如 LoFTR [9]），dense warp 方法在 coarse feature map 上计算全局对应（COTR [17]），另一些方法如 PATS [21]、OETR/Overlap Estimation [22,23] 和其他两阶段区域方法 [24] 则先预测区域/重叠区域再执行精细点匹配。

**原文指出的缺点：**  
　　这些 coarse-to-fine 方法的关键限制在于：

1. 中间 coarse 搜索空间往往是 patch-level 或重叠区域，**与图像语义没有明确关联**；
2. 一些重叠估计方法只能得到粗粒度区域，无法支持精细定位；
3. patch-level coarse 空间依然需要进行大量比较，计算代价高；
4. coarse 层分辨率较低，导致 fine 层无法恢复高精度对应。

**本文的改进方向：**  
　　本文设计了新的中间 coarse 搜索空间：

+ **语义区域（SOA + SIA）** —— 引入语义先验，使 coarse 空间具有更高的稳定性与语义一致性；
+ **GAM（Geometry Area Matching）** —— 在语义区域内部加入几何一致性检查（通过基本矩阵和 Sampson 残差），弥补语义化区域可能带来的歧义，实现可靠的区域匹配；
+ 最终构建 **A2PM（Area-to-Point Matching）** 结构，在语义区域指引下进行精细点匹配，既减少粗层冗余，也提高了最终点匹配的精度。

<details class="lake-collapse"><summary id="u81b6b00a"><strong><span class="ne-text" style="font-size: 16px">相关工作文献</span></strong></summary><p id="u77dc0d4b" class="ne-p"><span class="ne-text" style="font-size: 16px">[7] D. DeTone, T. Malisiewicz, and A. Rabinovich, SuperPoint: Self-Supervised Interest Point Detection and Description, CVPR Workshops, 2018.<br /></span><span class="ne-text" style="font-size: 16px">[8] Y. Ono, E. Trulls, P. Fua, and K. M. Yi, LF-Net: Learning Local Features from Images, NeurIPS, 2018.<br /></span><span class="ne-text" style="font-size: 16px">[9] J. Dusmanu, M. Rocco, T. Pajdla et al., D2-Net: A Trainable CNN for Joint Description and Detection of Local Features, CVPR, 2019.<br /></span><span class="ne-text" style="font-size: 16px">[10] J. Revaud, P. Weinzaepfel, C. de Souza, R2D2: Reliable and Repeatable Detector and Descriptor, NeurIPS, 2019.<br /></span><span class="ne-text" style="font-size: 16px">[11] J. Tyrol, G. Berton, Y. Zhong et al., DISK: Learning Local Features with Policy Gradient, NeurIPS, 2020.<br /></span><span class="ne-text" style="font-size: 16px">[12] J. Luo, S. Shen, L. Zhou, ASLFeat: Learning Local Features with Adaptive Sparse Layers, CVPR, 2020.<br /></span><span class="ne-text" style="font-size: 16px">[13] J. Zhang, Y. Sun, W. Hu et al., Learning to Remove Outliers for Robust Correspondence Estimation (OANet), ICCV, 2019.<br /></span><span class="ne-text" style="font-size: 16px">[14] E. Brachmann and C. Rother, Neural-Guided RANSAC (NG-RANSAC), ICCV, 2019.<br /></span><span class="ne-text" style="font-size: 16px">[15] V. Rocco, M. Arandjelović, J. Sivic, Efficient Neighbourhood Consensus Networks via Multi-scale Patch Matching (PATS), CVPR, 2021.<br /></span><span class="ne-text" style="font-size: 16px">[16] Z. Tang and K. Wang, DKM: Dense Kernelized Matching, CVPR, 2022.<br /></span><span class="ne-text" style="font-size: 16px">[17] J. Yang, D. Dai, L. Van Gool, COTR: Correspondence Transformer for Matching Across Images, ICCV, 2021.<br /></span><span class="ne-text" style="font-size: 16px">[19] Z. Wang, J. Zhang, W. Liu, QuadTree Attention for High-Resolution Feature Matching, ECCV, 2022.<br /></span><span class="ne-text" style="font-size: 16px">[20] J. Li, S. Bai, X. Bai, ASpanFormer: Local Feature Matching with Adaptive Span Transformer, ECCV, 2022.<br /></span><span class="ne-text" style="font-size: 16px">[21] V. Rocco, M. Arandjelović, J. Sivic, Efficient Neighbourhood Consensus Networks via Multi-scale Patch Matching (PATS), CVPR, 2021.<br /></span><span class="ne-text" style="font-size: 16px">[22] M. Chen, T. Zhang, S. Zhang, OETR: Overlap Estimation for Two-stage Robust Matching, CVPR, 2023.<br /></span><span class="ne-text" style="font-size: 16px">[23] S. Zhang, Y. Wang, L. Dai, OverlapFormer: Two-stage Feature Matching via Overlap Estimation, ICCV, 2023.<br /></span><span class="ne-text" style="font-size: 16px">[24] A. Schuster, M. Rothermel, F. Fraundorfer, Robust Two-stage Matching via Co-visible Region Detection, ECCV, 2022.  </span></p></details>
## ２论文方法
### A2PM框架
<!-- 这是一张图片，ocr 内容为：PM GR PLAS)>P(AS) 10 PM DETECT PM ISI PM 11 ASS AS GMC GP OUTPUT INPUT GEOMETRY AREA MATCHING SEMANTIC AREA MATCHING -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764401666017-c9c262c6-3aae-4cbf-8069-d664d91b181c.png)

 **A2PM 就是一个把“区域匹配 AM”和“点匹配 PM”组合在一起的框架。  **

 论文首先把 A2PM 定义为一个从图像对到最终点匹配的映射：  

<!-- 这是一张图片，ocr 内容为：PMA(IO,II,AM,PM). (1) -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764402288583-08e3f268-e517-4b9b-9881-bd9d909a5662.png)

<!-- 这是一张图片，ocr 内容为：1.输入是图像对IO,11. 2.A2PM需要两个关键模块: 区域匹配方法AM 点匹配方法PM 3.输出是最终的点匹配集合P三(GM,PM))M_I. 一这一步的核心:A2PM将"区域匹配"与"点匹配"组合在一起,从区域到点. -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764402420994-cd9bae6b-388b-421e-9608-3c10cd707e6e.png)

**（1）输入要求：两张具有重叠部分的 RGB 图像。**

不要求光照一致、不要求纹理丰富、不要求内参一致。

**（2）图像必须能做语义分割，以生成稳定的 SOA/SIA 区域。**

只需大体结构能分出来即可（SEEM-T 就能满足）。

**（3）SGAM 对输入图像的视觉质量要求不高，但必须有场景结构与语义信息。**

模糊/弱纹理/重复纹理场景也能很好处理。

 A2PM 的第一阶段——区域匹配  

<!-- 这是一张图片，ocr 内容为：区域匹配器AM会从两个图像中分别检测区域集合{G}与{8]8],并执行区域级匹配: {AIN(G } AM(IO,I1). 逻辑含义: 1.每个区域匹配是一个二元组A((A)(AI,B.(AI,B.(A). 2.(I)是指数映射,把IO的区域对应到I的匹配区域. 3,最终得到N个区域匹配对. 这一步建立了COARSE-LEVEL的"中间搜索空间". -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764402498936-bd71aebf-5eca-497b-8fc1-bca1d9890b56.png)

 A2PM 的第二阶段——在每个区域内做点匹配

 <!-- 这是一张图片，ocr 内容为：得到区域匹配之后,就在每一对区域内部执行点匹配PM: P(PM(AIN (3) 逻辑含义: 1.每个区域对A, 2.PM在该区域图像对上运行,得到区域内部的点对应集合: P{PM(AIM((((A)) 3.将所有区域内部得到的点匹配集合拼接起来一得到完整点匹配P. 二这里A2PM体现出它的核心思想;点匹配不是直接在整幅图上做,而是在"区域图像上做",从而是开稳定 性,减少冗余,提高分辨率. -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764402547765-8e637e3a-a2fc-4956-9f62-5b53c713828c.png)

### <font style="color:rgb(0, 0, 0);">SAM</font>
       为实现图像间的语义区域匹配，论文提出**语义区域匹配（SAM）**方法，论文首先定义了两个典型语义区域，旨在实现语义与搜索空间的深度融合。第一个区域是物体中心区域，称为**语义物体区域（SOA）**，其内部物体的纹理表面和显著边缘有利于点匹配。然而，某些物体（如靠近相机的物体）在图像中尺寸过大，导致对应区域的尺寸或长宽比异常膨胀，造成搜索空间失真。为此，我们进一步提出**语义交集区域（SIA）**，该区域由多个物体的交集部分构成，而非单一物体，从而高效捕捉大物体交集处的实体特征。

**SAM的总体流程**

+ 对两张图做 **语义分割**（得到每个像素的语义标签）。
+ 在每张图上检测出两类语义区域：**SOA（物体区域）** 和 **SIA（交叉区域）**。
+ 为每个区域计算一个 **描述子**（descriptor）：SOA 用**二值**“语义周边描述子”，SIA 用**实数**“语义比例描述子”。
+ 对两个图的区域做 **最近邻匹配**（SOA 用 Hamming，SIA 用 L2）。
+ 根据距离阈值筛掉显然坏的匹配；把“若干相近候选”标记为 **doubtful（疑似）**；把清楚的匹配记为 putative matches（初步匹配）。
+ 把 putative matches 和 doubtful areas 传给后续的 GAM（几何校验）进一步确认或剔除。

####  SOA：Semantic Object Area（检测 → 描述 → 匹配）  
<!-- 这是一张图片，ocr 内容为：SEMANTIC OBJECT AREA SEM.SURROUNDING DESCRIPTOR 01101001 SEM.SURROUNDINGS CONNECTED COMPONENT AREA DETECTION AREA DESCRIPTION INPUT LMAGE -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764403066471-eb493fa5-18a0-4b05-aa16-9940f1210e9f.png)

**       SOA 使用周边语义的二值描述子（semantic surrounding descriptor），用 Hamming 距离做最近邻匹配；不确定的候选被标注为 ‘doubtful’ 交给几何模块判断。  **

**1) SOA 的检测（如何得到物体区域）**

+ 输入：语义分割结果（每个像素的类别标签，例如 chair, table, floor 等）。本文直接采用预训练的语义分割模型**（如SEEM-L/T和Mask2Former）**对图像进行语义分割，获得高质量的语义类别标签图和实例级 Mask，为后续 SOA/SIA 区域检测提供基础信息。  
+ 步骤：
    1. 对每个语义类别取 **连通分量**（connected components）。每个连通分量就是一个 SOA 的初步候选（即一块连续的同类区域）。

```python
def static_connected_area_upspeed(self, sem, label_list, name=0):  
    """ 使用OpenCV进行连通区域分析，提取语义对象的连通分量  
        这是SOA检测的核心步骤，用于从语义分割图中识别连续的同类区域  
          
        Args:  
            sem: 输入的语义分割图像，每个像素值为语义类别ID  
            label_list: 需要处理的语义标签列表  
            name: 图像标识（0或1），用于调试输出  
              
        Returns:  
            sem_connect_dict = {  
                label: [[connect_area]s]  # 每个标签对应多个连通区域  
                    connect_area: [[u, v]s]  # 每个连通区域的像素坐标列表  
            }  
        """  
    logger.info(f"static connected area starting...")  
    sem_connect_dict = {}  
  
    # 遍历每个语义标签，分别提取其连通区域  
    for label in label_list:  
        # 步骤1: 为当前标签创建二值图像  
        # 将属于当前标签的像素设为255，其他设为0  
        temp_bin_img = self._get_bin_img(sem, label, name)  
          
        # 步骤2: 使用OpenCV的连通分量分析算法  
        # 返回连通区域数量和标记图像（每个区域有唯一ID）  
        temp_connect_num, connect_label_img = cv2.connectedComponents(temp_bin_img)  
          
        # 初始化当前标签的连通区域列表  
        if label not in sem_connect_dict.keys():  
            sem_connect_dict.update({label:[]})  
              
        # 步骤3: 遍历所有连通区域（从1开始，0是背景）  
        for connect_label_id in range(1, temp_connect_num+1):  
            # 获取当前连通区域的所有像素坐标  
            temp_where_set = np.where(connect_label_img == connect_label_id)  
            N = temp_where_set[0].shape[0]  # 连通区域的像素数量  
              
            # 步骤4: 过滤过小的区域  
            # 如果区域像素数小于阈值，跳过该区域（避免噪声干扰）  
            if N < self.connected_thd:  
                continue  
                  
            # 步骤5: 将numpy的where结果转换为坐标列表格式  
            # 从(array([v]), array([u]))转换为[[v, u], [v, u], ...]  
            area_coord_list = self._convert_where_to_uvlist(temp_where_set)  
              
            # 将处理后的连通区域添加到结果字典中  
            sem_connect_dict[label].append(area_coord_list)  
  
    return sem_connect_dict
```

    2. 对空间上非常接近、且语义相同的小块进行 **合并（merge）**，以避免把同一个物体分成多个区域（例如被遮挡或分割的小片段）。

```python
def combine_single_label_patches(self, patches, sem):  
    """  
        合并具有相同语义标签的相邻区域块  
        这是SOA处理中的关键步骤，用于避免同一物体被分割成多个小片段  
          
        workflow:  
            1. 计算所有区域的中心点  
            2. 以第一个区域为主，计算与其他区域的中心距离  
            3. 对距离排序，找到最近的区域  
            4. 根据距离阈值决定是合并还是分离  
              
        Args:  
            patches: 待合并的区域列表，格式为 [[u_min, u_max, v_min, v_max, label, desc], ...]  
            sem: 语义分割图像，用于重新构建合并区域的描述子  
              
        Returns:  
            combined patches: 合并后的区域列表，格式与输入相同  
        """  
    # 复制输入列表，避免修改原始数据  
    combined_patches = patches[:]  
    # 存储无法合并的独立区域  
    cannot_combine_patches = []  
    logger.info(f"got {len(combined_patches)} patches to combinne")  
  
    # 循环处理直到只剩一个区域或没有区域  
    while (len(combined_patches)>1):  
        # 步骤1: 计算所有区域的中心点坐标  
        centers = []  
        centers = [[(x[0]+x[1])/2 , (x[2]+x[3])/2] for x in combined_patches]  
          
        # 步骤2: 计算第一个区域与其他所有区域的欧氏距离  
        dists = []  
        for i in range(1, len(centers)):  
            dists.append(calc_euc_dist_2d(centers[0], centers[i]))  
            logger.info(f"got dist = {dists}")  
          
        # 步骤3: 对距离排序，找到最近的区域  
        dists_sorted = sorted(dists)  
          
        # 步骤4: 根据距离阈值决定合并或分离  
        if dists_sorted[0] > self.combined_obj_dist_thd:  
            # 如果最近距离仍大于阈值，说明第一个区域是独立的  
            cannot_combine_patches.append(combined_patches.pop(0))  
        else:  
            # 找到最近区域的索引，进行合并  
            close_idx = dists.index(dists_sorted[0])  
            # 合并两个相邻区域  
            combined_patch_temp = self.combine_two_patches(combined_patches.pop(0), combined_patches.pop(close_idx), sem)  
            # 将合并后的区域放回列表开头，继续下一轮处理  
            combined_patches.insert(0, combined_patch_temp)  
  
        logger.info(f"After combination got {len(combined_patches)} + {len(cannot_combine_patches)} patches")  
  
    # 返回合并后的区域和独立区域的并集  
    return combined_patches + cannot_combine_patches
```

    3. 记录每个 SOA 的边界框（bounding box）、面积、质心等几何信息，方便后续裁剪与描述。

**2) SOA 的描述子（Semantic Surrounding Descriptor，二值向量）**

目标：区分同一语义类别的不同实例（例如图片里有两个椅子），利用**周围的语义上下文**来区分。

+ 描述子结构：
    - 一个二进制向量，每一位对应一个语义类别（例如：桌子、地板、墙、窗户……共有 C 类），或者更常见的是“类别 × 多尺度边界位置”的组合（见下）。
    - 如果在 SOA 的**边界周围**（或扩展的多尺度边界）检测到某个语义出现，则该位为 1；否则为 0。
+ 具体实现要点：
    1. 在 SOA 边界外按多个尺度做 **multiscale boundary capture**（如边界外扩 1×、1.5×、2×），在这些环带上统计出现的语义类别。
    2. 每个尺度产生一个二值向量（该语义是否出现在该尺度的周边），把多尺度向量拼接或合并（如按位 OR），得到最终的二值描述子。
    3. 为提高区分度，也可把边界按方向/区段划分（例如 8 个方向），每方向做同样统计 → 更丰富但也更长的二值向量。
+ 优点：对“周围长什么”敏感，能把“两个看起来类似的椅子但周围有不同物体”区分开来。

```python
def construct_bin_desc_with_order_along_bound_multiscale(self, area, label, sem, radius=2, ms_list=[1, 1.6, 2.2]):  
    """ 构建多尺度边界二值描述子，用于SOA匹配  
        通过在不同尺度上捕获对象边界周围的语义分布来区分同类别的不同实例  
          
        Args:  
            area: 对象区域边界框 [u_min, u_max, v_min, v_max]  
            label: 当前对象的语义标签  
            sem: 语义分割图像  
            radius: 边界采样半径  
            ms_list: 多尺度列表，默认为[1, 1.6, 2.2]  
              
        Returns:  
            二值描述子列表，长度为label_size*4（4个方向）  
              
        图示说明：  
           -----------  # 原始对象区域  
           | ------  |  # 1.6倍扩展区域  
           | |    |  |  # 1.0倍扩展区域    
           | |    |  |  
           | ------  |  
           -----------  
        """  
  
    # 初始化最终描述子向量：4个方向×标签数量，初始值为False  
    desc_final_list = np.zeros((self.label_size*4)) == 1  
    W, H = self.size  
  
    # 解包区域边界坐标  
    u_min, u_max, v_min, v_max = area  
  
    # 计算原始区域的半径和中心点  
    raw_u_r = (u_max - u_min) // 2  # 水平半径  
    raw_v_r = (v_max - v_min) // 2  # 垂直半径  
  
    raw_u_center = (u_max + u_min) // 2  # 水平中心  
    raw_v_center = (v_max + v_min) // 2  # 垂直中心  
  
    # 遍历所有尺度，构建多尺度描述子  
    for scale in ms_list:  
        # 根据尺度因子计算扩展后的半径  
        ms_u_r = raw_u_r * scale  
        ms_v_r = raw_v_r * scale  
          
        # 计算扩展后的区域边界，确保不超出图像范围  
        u_max_ms = min(int(raw_u_center + ms_u_r), W-radius)  
        u_min_ms = max(int(raw_u_center - ms_u_r), 0)  
        v_max_ms = min(int(raw_v_center + ms_v_r), H-radius)  
        v_min_ms = max(int(raw_v_center - ms_v_r), 0)  
  
        # 确定哪些边界需要处理  
        # bound_label = [上边界, 右边界, 下边界, 左边界]  
        # 1表示需要处理，0表示不需要（当扩展边界与原始边界重合时）  
        bound_label = [1,1,1,1]  
        if v_min_ms == v_min:  # 上边界重合  
            bound_label[0] = 0  
            if u_max_ms == u_max:  # 右上角重合  
                bound_label[1] = 0  
        if v_max_ms == v_max:  # 下边界重合  
            bound_label[2] = 0  
            if u_min_ms == u_min:  # 左下角重合  
                bound_label[3] = 0  
                  
        # 为当前尺度构建方向性边界描述子  
        temp_desc = self.construct_bin_desc_with_order_along_bound([u_min_ms, u_max_ms, v_min_ms, v_max_ms], label, bound_label, sem, radius) == 1  
          
        # 将当前尺度的描述子与最终描述子进行OR操作（合并语义信息）  
        desc_final_list = desc_final_list | temp_desc  
  
    # 将布尔数组转换为整数列表并返回  
    return (desc_final_list * 1).tolist()
```

**3) SOA 的匹配（Nearest Neighbor + Hamming distance）**

+ 对两张图中所有 SOA 的二值描述子做**最近邻检索**（通常是 Hamming 距离，因为描述子是二值）。
+ 匹配策略：
    1. 对每个 SOA（源图）找到目标图中 **Hamming 距离最小** 的候选 SOA。

```python
def _compare_obj_desc_bin(self, desc0, desc1):  
    """ 计算两个二值描述子之间的汉明距离  
        这是SOA匹配中的核心比较函数，用于衡量语义对象描述子的相似性  
          
        Args:  
            desc0: 第一个二值描述子，格式为 [Nx1] 的numpy数组  
            desc1: 第二个二值描述子，格式为 [Nx1] 的numpy数组  
                   描述子是通过多尺度边界采样构建的，包含对象周围的语义信息  
          
        Returns:  
            h_loss: 汉明损失值，范围在[0,1]之间  
                   0表示两个描述子完全相同，1表示完全不同  
                   在SOA匹配中，1 - h_loss 用作相似度度量  
          
        Note:  
            - 汉明距离计算两个等长字符串在相同位置上不同字符的个数  
            - 对于二值描述子，汉明距离等于不同位的数量除以总位数  
            - 该函数在_find_match_obj()中被调用，用于筛选匹配的语义对象  
        """  
    # 使用sklearn的hamming_loss函数计算汉明距离  
    # 该函数自动处理输入验证和标准化计算  
    h_loss = hamming_loss(desc0, desc1)  
    return h_loss
```

    2. 若最小距离 > 阈值 **TH** → 直接 **拒绝**（显然不是匹配）。
    3. 若存在多个候选距离相近（差距小于阈值 **Tda**），把这个 SOA 标记为 **doubtful（疑似）**，交给后续的 GAM Predictor 进一步判断。
    4. 否则把该匹配加入 putative area matches（初步匹配集合）。
+ 关键阈值：
    - **TH**：最大接受距离阈值（若距离太大，认为不是同语义实例的匹配）
    - **Tda**：疑似阈值（如果多个候选距离差小于 Tda，则标记为 doubtful）

**Th（0.5）：用于拒绝 SOA 匹配。**  
如果语义描述子的海明距离太大，则匹配不成立。

**Tda（0.2）：用于发现可疑匹配。**  
当多个候选区域的语义描述子距离非常接近时，将该区域标记为“doubtful”，后续由几何匹配器再判断。

这两个阈值均为经验设置。

| 区域类型 | 定义 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **SOA** （语义物体区域） | 单个语义类别对应的完整物体，如桌子、墙、椅子 | 大区域、语义信息强、检测简单 | 太大时特征单一（如整面墙）→ 容易歧义 |
| **SIA** （语义交叉区域） | 多个 SOA 的交界区域组合 | 特征结构明显、形状独特 → 匹配能力强 | 需要额外处理才能生成 |


####  SIA：Semantic Intersection Area（检测 → 描述 → 匹配）  
<!-- 这是一张图片，ocr 内容为：SEMANTIC INTERSECTION AREA SEM.PROPOTION DESAIPTOR DETECTION TOP LEVEL 00000.00201020.10101 SEM.PROPOTION BOTTOM LEVEL REFINEMENT (U,) ARGMINGMING 0.1 MINA(U,V) TO(O.I A10 INPUT LMAGE AREA DETECTION AREA DESCRIPTION -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764403081627-7c10069f-ac16-4dca-8214-72fba2cb54ae.png)

**1) SIA 的检测（滑窗 + two-layer semantic pyramid）**

SIA 不是语义分割直接给出的，而是通过滑窗找“多语义混合”的小区域。

+ 目标：找到包含 **≥3 个不同语义** 的局部窗口（说明该窗口处在多个物体交界处，结构信息丰富）。
+ 直接滑窗很慢，论文采用 **two-layer semantic pyramid** 来加速：
    1. **Top layer（缩小尺度）**：把语义图下采样（例如缩小 r 倍），窗口也按该尺度缩小，用较大步长滑窗做粗检测 → 快速筛出可能位置。

```python
def _stastic_overlap_candis_list(self, sem, black_list_label, window_radius, resize_ratio):  
    """ 在语义分割图上统计重叠区域候选位置  
        这是SIA检测的核心步骤，通过滑窗寻找包含多种语义的交界区域  
          
        Args:  
            sem: 输入的语义分割图像，每个像素值为语义类别ID  
            black_list_label: 需要排除的背景标签  
            window_radius: 滑窗半径，控制检测区域大小  
            resize_ratio: 缩放比例，用于金字塔层处理  
              
        Returns:  
            sem_overlap_centers: 检测到的重叠区域中心点列表 [[u, v], ...]  
            sem_overlap_labels: 对应的语义标签字符串列表 ["label1_label2_label3", ...]  
            sem_overlap_vars: 对应的语义分布方差列表  
        """  
    H, W = sem.shape  
    # 根据缩放比例调整窗口半径  
    window_radius = int(window_radius * resize_ratio)  
    logger.info(f"stastic with radius = {window_radius} in {W}x{H} sem")  
      
    # 初始化结果列表  
    sem_overlap_centers = []  
    sem_overlap_labels = []  
    sem_overlap_vars = []  
  
    # 遍历图像中的每个像素位置（留出边界）  
    for u in range(window_radius, W-window_radius):  
        for v in range(window_radius, H-window_radius):  
            # 检查当前点是否为有效的重叠区域候选  
            flag, label_str, variance = self._stastic_single_point(sem, u, v, window_radius, black_list_label, resize_ratio)  
            if not flag:   
                # 如果不是有效候选，跳过  
                continue  
  
            # 如果是第一个候选，直接添加  
            if len(sem_overlap_centers) == 0:  
                sem_overlap_centers.append([u, v])  
                sem_overlap_labels.append(label_str)  
                sem_overlap_vars.append(variance)  
            else:  
                # 计算与已有候选的距离  
                dists = [math.sqrt((u-c[0])**2 + (v-c[1])**2) for c in sem_overlap_centers]  
                copy_centers = sem_overlap_centers[:]  
                # 初始化融合候选列表  
                fuse_candi_centers = [[u,v]]  
                fuse_candi_labels = [label_str]  
                fuse_candi_vars = [variance]  
  
                # 收集距离过近的候选（需要合并）  
                for idx_dist, dist in enumerate(dists):  
                    if dist < self.same_overlap_dist * resize_ratio:  
                        pop_obj = copy_centers[idx_dist]  
                        pop_idx = sem_overlap_centers.index(pop_obj)  
                        fuse_candi_centers.append(sem_overlap_centers.pop(pop_idx))  
                        fuse_candi_labels.append(sem_overlap_labels.pop(pop_idx))  
                        fuse_candi_vars.append(sem_overlap_vars.pop(pop_idx))  
  
                # 如果没有需要合并的候选，直接添加  
                if len(fuse_candi_centers) == 1:  
                    sem_overlap_centers.append([u, v])  
                    sem_overlap_labels.append(label_str)  
                    sem_overlap_vars.append(variance)  
                else:  
                    # 多个候选需要合并，进行筛选  
                      
                    # 第一轮筛选：优先选择语义标签数量最多的候选  
                    label_len_candi_centers = []  
                    label_len_candi_labels = []  
                    label_len_candi_vars = []  
  
                    # 计算每个候选的语义标签数量  
                    label_lens = [len(label.split('_')) for label in fuse_candi_labels]  
                    max_len = max(label_lens)  
  
                    # 收集标签数量最多的候选  
                    for i, label in enumerate(fuse_candi_labels):  
                        if len(label.split("_")) == max_len:  
                            label_len_candi_centers.append(fuse_candi_centers[i])  
                            label_len_candi_labels.append(fuse_candi_labels[i])  
                            label_len_candi_vars.append(fuse_candi_vars[i])  
                      
                    # 如果只有一个候选标签数量最多，直接选择  
                    if len(label_len_candi_centers) == 1:  
                        sem_overlap_centers.append(label_len_candi_centers[0])  
                        sem_overlap_labels.append(label_len_candi_labels[0])  
                        sem_overlap_vars.append(label_len_candi_vars[0])  
                    else:  
                        # 第二轮筛选：在标签数量相同的情况下，选择方差最小的候选  
                        min_var = min(label_len_candi_vars)  
                        min_var_idx = label_len_candi_vars.index(min_var)  
  
                        sem_overlap_centers.append(label_len_candi_centers[min_var_idx])  
                        sem_overlap_labels.append(label_len_candi_labels[min_var_idx])  
                        sem_overlap_vars.append(label_len_candi_vars[min_var_idx])  
                      
    return sem_overlap_centers, sem_overlap_labels, sem_overlap_vars
```

    2. **Bottom layer（原始尺度）**：在候选位置回到原图做精细微调（refinement）。

```python
def _refine_overlap_in_ori_sem_list(self, sem, pyramid_overlap_centers, pyramid_overlap_labels, pyramid_ratio, black_list_label):  
    """ 在原始分辨率语义图上精细调整重叠区域中心位置  
        这是SIA检测金字塔方法的bottom layer步骤，对top layer检测到的候选进行精确定位  
          
        Args:  
            sem: 原始分辨率的语义分割图像  
            pyramid_overlap_centers: 金字塔层检测到的候选中心点列表 [[u, v], ...]  
            pyramid_overlap_labels: 对应的语义标签字符串列表  
            pyramid_ratio: 金字塔缩放比例（如1/8）  
            black_list_label: 需要排除的背景标签  
              
        Returns:  
            ori_overlap_centers: 精细化后的中心点列表 [[u, v], ...]  
            ori_overlap_labels: 对应的语义标签列表  
        """  
    # 初始化输出列表  
    ori_overlap_centers = []  
    ori_overlap_labels = []  
    H, W = sem.shape  
    # 计算原始分辨率下的搜索半径（金字塔比例的倒数）  
    ori_radius = int(1/pyramid_ratio)  
  
    # 遍历金字塔层检测到的每个候选中心  
    for i, [u, v] in enumerate(pyramid_overlap_centers):  
        # 步骤1: 将金字塔坐标转换到原始分辨率坐标  
        u_ori = int(u / pyramid_ratio)  
        v_ori = int(v / pyramid_ratio)  
        # 确保中心点不超出图像边界  
        u_ori, v_ori = self._refine_center(u_ori, v_ori, ori_radius, W, H)  
          
        # 步骤2: 定义局部搜索区域  
        u_s = max(0, u_ori-ori_radius)  # 搜索区域左边界  
        u_e = min(W, u_ori+ori_radius)  # 搜索区域右边界  
        v_s = max(0, v_ori-ori_radius)  # 搜索区域上边界  
        v_e = min(H, v_ori+ori_radius)  # 搜索区域下边界  
        label_str_ori = pyramid_overlap_labels[i]  
  
        # 步骤3: 在局部区域内搜索方差最小的最优中心  
        min_var = 1e5  # 初始化最小方差为大值  
        min_var_u = u_ori  # 最优中心坐标  
        min_var_v = v_ori  
        min_var_label = label_str_ori  
          
        # 在搜索区域内以步长2进行精细搜索（提高效率）  
        for u_ in range(u_s, u_e, 2):  
            for v_ in range(v_s, v_e, 2):  
                # 检查当前位置是否为有效的重叠区域  
                flag, label_str, var = self._stastic_single_point(sem, u_, v_, self.overlap_radius, black_list_label, 1)  
                if not flag: continue  
                # 如果当前位置的语义分布方差更小，更新最优位置  
                if var < min_var:  
                    min_var = var  
                    min_var_u = u_  
                    min_var_v = v_  
                    min_var_label = label_str  
  
        # 如果没有找到有效位置，跳过该候选  
        if min_var == 1e5: continue  
  
        # 将找到的最优位置添加到结果列表  
        ori_overlap_centers.append([min_var_u, min_var_v])  
        ori_overlap_labels.append(min_var_label)  
          
    return ori_overlap_centers, ori_overlap_labels
```

+ 在精细层，为了让 SIA 更“均匀”且更有代表性，论文会**调整窗口中心**（在窗口内按一定范围搜索）以最小化该窗口内 **语义比例的方差 σ(u,v)**（目标是使窗口里的语义分布更均匀/稳定）。

**2) SIA 的描述子（semantic proportion descriptor，实数向量）**

+ 描述子是 **每个语义类别在该窗口内所占的比例**（例如：chair 0.3, floor 0.5, wall 0.2 ……），这是一个实数向量（长度为类别数 C，一般大部分值为 0）。

```python
def _expand_area_from_center_form_desc(self, sem, overlap_centers, obj_centers, obj_scales, std_radius, black_list_label):  
    """ 从中心点扩展重叠区域并构建语义比例描述子  
        这是SIA检测的最后步骤，将检测到的中心点扩展为实际区域并构建描述子  
          
        Args:  
            sem: 输入的语义分割图像  
            overlap_centers: 重叠区域候选中心点列表 [[u, v], ...]  
            obj_centers: 已检测到的对象区域中心点列表（用于尺度参考）  
            obj_scales: 对应的尺度因子列表  
            std_radius: 标准重叠区域半径  
            black_list_label: 需要排除的背景标签  
              
        Returns:  
            overlap_areas: 扩展后的重叠区域边界框列表 [[u_min, u_max, v_min, v_max], ...]  
            overlap_area_descs: 对应的语义比例描述子列表  
        """  
    overlap_areas = []  
    overlap_area_descs = []  
    W, H = self.size  
  
    # 遍历每个重叠区域候选中心点  
    for center in overlap_centers:  
        # 步骤1: 确定自适应尺度  
        scale = 1.0  # 默认尺度  
        u_c, v_c = center  
          
        # 如果存在已检测的对象，使用最近对象的尺度作为参考  
        if len(obj_centers) > 0:  
            # 计算当前中心点到所有对象中心的欧氏距离  
            dists = [math.sqrt((u_c-obj_center[0])**2 + (v_c - obj_center[1])**2) for obj_center in obj_centers]  
            min_dist = min(dists)  
            min_idx = dists.index(min_dist)  
  
            # 如果距离阈值内，采用最近对象的尺度  
            if min_dist < self.same_overlap_dist:  
                scale = obj_scales[min_idx]  
          
        # 根据尺度计算实际半径  
        radius = std_radius * scale  
  
        # 步骤2: 精确调整中心点位置，确保区域不超出图像边界  
        u_t, v_t = self._refine_center(u_c, v_c, radius, W, H)  
  
        # 步骤3: 计算区域边界框  
        u_min = int(u_t - radius)  
        u_max = int(u_t + radius)  
        v_min = int(v_t - radius)  
        v_max = int(v_t + radius)  
  
        # 将边界框添加到结果列表  
        overlap_areas.append([u_min, u_max, v_min, v_max])  
  
        # 步骤4: 构建语义比例描述子  
        # 提取区域内的语义分割图像块  
        sub_sem = sem[v_min:v_max, u_min:u_max]  
        # 将图像块转换为一维列表  
        temp_patch = np.squeeze(sub_sem.reshape((-1,1))).tolist()  
        # 统计每个语义标签的像素数量  
        temp_stas_dict = collections.Counter(temp_patch)  
  
        # 计算区域总面积（用于归一化）  
        total_valid_size = (radius*2)**2  
        sorted_label_list = sorted(self.label_list)  
        l_total = len(sorted_label_list)  
        # 初始化描述子向量  
        desc_temp = np.zeros((l_total, 1))  
  
        # 为每个出现的语义标签计算比例  
        for label in temp_stas_dict.keys():  
            if label == black_list_label: continue  # 跳过背景标签  
            # 计算该标签在区域内的比例（像素数/总面积）  
            desc_temp[sorted_label_list.index(label),0] = temp_stas_dict[label] / total_valid_size  
          
        # 将描述子添加到结果列表  
        overlap_area_descs.append(desc_temp)  
  
    return overlap_areas, overlap_area_descs
```

+ 为增强尺度鲁棒性，在同一中心上使用 **多尺度窗口**（不同大小），计算每尺度的比例向量，再取平均得到最终 SIA 描述子。

**3) SIA 的匹配（Nearest Neighbor + L2 distance）**

+ 对两个图里所有 SIA 的实数比例向量做 **L2 最近邻检索**（欧氏距离）。

```python
def match_overlap_area_pyramid_version(self):  
    """ 实现重叠区域的双向匹配流程  
        这是SIA（Semantic Intersection Area）匹配的核心方法，处理包含多种语义的交界区域  
          
        Returns:  
            self.matched_overlap_label: 匹配的重叠区域标签列表  
            self.matched_overlap_area0: 图像0中匹配的重叠区域边界框列表  
            self.matched_overlap_area1: 图像1中匹配的重叠区域边界框列表  
        """  
    # 步骤1: 检测重叠区域并构建描述子  
    # 对两个图像分别调用金字塔方法检测重叠区域  
    self.sem0_overlap_areas, self.sem0_overlap_area_descs = self.achieve_overlap_area_pyramid_main(  
        self.sem0, self.color0, self.obj_centers0, self.obj_scale0,   
        pyramid_ratio=1/8, name="pyramid_sem0_overlap"  
    )  
    self.sem1_overlap_areas, self.sem1_overlap_area_descs = self.achieve_overlap_area_pyramid_main(  
        self.sem1, self.color1, self.obj_centers1, self.obj_scale1,   
        pyramid_ratio=1/8, name="pyramid_sem1_overlap"  
    )  
  
    # 步骤2: 初始化双向匹配结果  
    self.matched_overlap_label = []  
    self.matched_overlap_area0 = []  
    self.matched_overlap_area1 = []  
    overlap_desc_dist_thd = self.overlap_desc_dist_thd  # L2距离阈值  
    overlap0_len = len(self.sem0_overlap_areas)  
  
    # 步骤3: 0→1方向匹配  
    # 为图像0的每个区域找到图像1中的最佳匹配  
    temp_021_matched_idx0 = [-1]*overlap0_len  # 初始化匹配索引数组  
  
    for idx0, desc0 in enumerate(self.sem0_overlap_area_descs):  
        min_dist = 1e5  # 初始化最小距离为大值  
        min_dist_idx = -1  
        # 遍历图像1中的所有描述子，寻找最近邻  
        for i, desc1 in enumerate(self.sem1_overlap_area_descs):  
            temp_dist = np.linalg.norm(desc0 - desc1)  # 计算L2距离  
            if temp_dist < min_dist:  
                min_dist = temp_dist  
                min_dist_idx = i  
  
        # 如果最小距离超过阈值，拒绝该匹配  
        if min_dist > self.overlap_desc_dist_thd:  
            continue  
  
        temp_021_matched_idx0[idx0] = min_dist_idx  
  
    # 步骤4: 1→0方向匹配  
    # 为图像1的每个区域找到图像0中的最佳匹配  
    temp_021_matched_idx1 = [-1]*overlap0_len  
  
    for idx1, desc1 in enumerate(self.sem1_overlap_area_descs):  
        min_dist = 1e5  
        min_dist_idx = -1  
        # 遍历图像0中的所有描述子，寻找最近邻  
        for idx0, desc0 in enumerate(self.sem0_overlap_area_descs):  
            temp_dist = np.linalg.norm(desc0 - desc1)  
            if temp_dist < min_dist:  
                min_dist = temp_dist  
                min_dist_idx = idx0  
  
        # 如果最小距离超过阈值，拒绝该匹配  
        if min_dist > self.overlap_desc_dist_thd:  
            continue  
  
        temp_021_matched_idx1[min_dist_idx] = idx1  
  
    # 步骤5: 双向一致性检查  
    # 只有当两个方向的匹配结果一致时才接受该匹配对  
    for idx0, idx1 in enumerate(temp_021_matched_idx0):  
        if idx1 != temp_021_matched_idx1[idx0] or idx1 == -1:   
            continue  # 跳过不一致或无效的匹配  
        self.matched_overlap_area0.append(self.sem0_overlap_areas[idx0])  
        self.matched_overlap_area1.append(self.sem1_overlap_areas[idx1])  
  
    logger.info(f"achieve {len(self.matched_overlap_area0)} matched overlap areas")  
  
    return self.matched_overlap_area0, self.matched_overlap_area1
```

+ 匹配策略与 SOA 类似：
    1. 找到 L2 距离最小候选。
    2. 若距离 > 阈值 **Tl** → 拒绝。
    3. 若多个候选距离相近（小于 Tda）→ 标注为 **doubtful**。
    4. 其余加入 putative matches。

**Tl（0.75）：用于 SIA 的匹配拒绝阈值，**SIA 语义比例有浮动，所以需要更宽松的拒绝阈值 Tl = 0.75。

### GAM
**为什么需要 GAM？**

因为 **SAM 只靠语义匹配**，会出现两个严重问题：

问题 1：语义区域之间描述子可能很像,语义描述子无法准确区分 → 出现 **doubtful areas**（歧义区域）。

问题 2：仅用语义没有几何关系,语义只知道“是什么东西”，但不知道：物体之间相对位置,投影几何关系,相机运动产生的对极几何。

因此：**语义匹配只能给出粗匹配，几何匹配必须来做最后判断。**

**GAM的核心思想：**

+ **在每个区域内部做点匹配（PM）**
+ **拟合基础矩阵 F**
+ **用 Sampson 距离验证几何一致性**

**GAM 利用几何一致性（对极几何 + Sampson 距离）来验证语义区域匹配，解决 SAM 的语义歧义问题。**

**GAM 是由 3 个模块组成的：**

**GP – Geometry Predictor（预测最可信的匹配组合）**

**GR – Geometry Rejector（用几何一致性淘汰假匹配）**

**GMC – Global Match Collection（补点匹配，防止区域太少）**

####   GP – Geometry Predictor（预测最可信的匹配组合）
**目标：**

+ 解决 SAM 标记为 doubtful 的区域，找到正确的匹配组合。

<!-- 这是一张图片，ocr 内容为：输入DOUBTFUL区域对 裁剪区域,统一尺度 生成所有区域组合AJ 取出下一个区域组合AJ 对AJ内所有区域对做点匹配 得到点集MJ 估计基础矩阵FJ 计算组合误差E(AJ) (SAMPSON 残差平均值) 还有组合? 选择误差最小的组合A* 输出A*中的最佳区域PAIR -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764491350828-7b5322fa-3cbc-4c8f-903a-54c2e3457d77.png)

**流程：**

1. **输入**
    - 两张图的 doubtful 区域（例如 SAM 标记的椅子/桌子/墙等歧义区域）
    - 每个区域内的语义描述子
    - 可选的初步点匹配方法（LoFTR、SuperPoint 等）
2. **核心操作**
    - 列出所有合理的区域组合候选（ 语义区域天然稳定，歧义区域极少且每个区域只有 2–3 个候选，GP 只需在极小的候选子集上枚举。）
    - 对每个组合：
        * 在每个区域内部执行点匹配 → 得到一组对应点

```python
def match_all_doubt_areas(self):  
    """ 根据匹配索引对所有doubtful区域执行点匹配  
        这是GP (Geometry Predictor)的核心步骤，为每种可能的匹配组合生成对应点  
          
        Args:  
            self.ori_doubt_areas0: 图像0中的doubtful区域列表，格式为 [u_min, u_max, v_min, v_max]  
            self.ori_doubt_areas1: 图像1中的doubtful区域列表，格式为 [u_min, u_max, v_min, v_max]  
            self.match_idx: 所有可能的匹配组合列表  
                格式为 [[(area_idx0, area_idx1), (...), (...)], [other match area situation]]  
                每个内层列表代表一种完整的匹配方案（所有区域的配对方式）  
                  
        Returns:  
            self.corrs_doubt_all: 每种匹配组合的对应点列表  
                格式为 [[[area_pair_corrs: list], ...], [other area match situation]]  
                外层列表对应不同的匹配方案，内层列表包含该方案中每对区域的对应点  
        """  
    # 初始化结果存储：按匹配方案组织对应点  
    self.corrs_doubt_all = []  
    logger.info(f"Got {len(self.match_idx)} match situations and each situation got {len(self.match_idx[0])} match pairs")  
      
    # 遍历每种匹配方案（situation）  
    for i, match_situation in enumerate(self.match_idx):  
        self.corrs_doubt_all.append([])  
          
        # 对当前方案中的每对区域执行点匹配  
        for j, match_pair in enumerate(match_situation):  
            idx0 = match_pair[0]  # 图像0中的区域索引  
            idx1 = match_pair[1]  # 图像1中的区域索引  
              
            # 在这对区域内执行点匹配，获取对应点  
            temp_corrs = self.match_area_pair_mind_size(  
                self.ori_doubt_areas0[idx0],   
                self.ori_doubt_areas1[idx1],   
                name=f"sub_corr_{i}_{j}"  
            )  
              
            # 存储这对区域的对应点  
            self.corrs_doubt_all[i].append(temp_corrs)  
              
            # 如果没有匹配到对应点，跳过可视化  
            if len(temp_corrs) == 0: continue  
              
            # 可视化选项：绘制匹配结果用于调试  
            if self.draw_verbose == 1:  
                self.draw_area_match_res(temp_corrs, f"matches_{i}_{j}")  
              
            logger.info(f"match for pair {idx0} in img0 and {idx1} in img1 done and get {len(temp_corrs)} correspondenses")  
  
    return self.corrs_doubt_all
```

        * 使用这些点拟合基础矩阵 F（ RANSAC + 8 点法  ）

```python
def calc_F(self, corrs):  
    """ 使用RANSAC算法估计基础矩阵F  
        这是几何验证的核心函数，用于计算两幅图像之间的对极几何关系  
          
        Args:  
            corrs: 点对应列表，格式为 [[u0, v0, u1, v1], ...]  
                  其中(u0, v0)是第一幅图像中的点，(u1, v1)是第二幅图像中的对应点  
                    
        Returns:  
            F: 3x3的基础矩阵，描述两幅图像间的对极几何约束  
               如果对应点数量不足则返回None  
                 
        Note:  
            - 基础矩阵满足 x2^T * F * x1 = 0 的对极约束  
            - 至少需要8个对应点才能估计基础矩阵（8点算法）  
            - 使用RANSAC去除外点，提高估计的鲁棒性  
        """  
    # 将输入转换为numpy数组格式  
    corrs = np.array(corrs)  
    corrs_num = len(corrs)  
      
    # 检查对应点数量是否满足最小要求  
    # 8点算法至少需要8个对应点来估计基础矩阵  
    if corrs_num < 8:   
        logger.warning(f"corrs num is {corrs_num}, too small to calc F")  
        return None  
      
    # 分离对应点：提取两幅图像中的点坐标  
    corrs_F0 = corrs[:, :2]  # 第一幅图像中的点 [u0, v0]  
    corrs_F1 = corrs[:, 2:]  # 第二幅图像中的点 [u1, v1]  
      
    logger.info(f"achieve corrs with shape {corrs_F0.shape} == {corrs_F1.shape} to calc F")  
      
    # 使用OpenCV的RANSAC算法估计基础矩阵  
    # FM_RANSAC: 使用RANSAC算法的8点算法  
    # ransacReprojThreshold=1: 重投影误差阈值（像素）  
    # confidence=0.99: RANSAC置信度  
    F, mask = cv2.findFundamentalMat(corrs_F0, corrs_F1,   
                                   method=cv2.FM_RANSAC,  
                                   ransacReprojThreshold=1,   
                                   confidence=0.99)  
      
    logger.info(f" calc F as \n {F}")  
  
    return F
```

        * 计算每个组合的 **Sampson 距离**（衡量几何一致性）

<!-- 这是一张图片，ocr 内容为：对区域内部的点匹配集合P三{(G")},估计基础矩阵E,定义 ,定义区域内部的SAMPSON距离累积为: M PIN F:QIN) (4) D(FI,P). (EG)(E明)(EDEDEPPP M1 逻辑含义: 1,根据区域内部的点对应估计F. 2.用SAMPSON距离衡量这些点对是否满足极线约束. 3,若区域内部匹配质量高,则 DIJI O -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764428773741-e8b31b7e-c57f-4a07-be4e-32182cae4eb2.png)

```python
def calc_geo_consistency_single_situ(self, single_situation_corrs):  
    """ 计算单一匹配组合的几何一致性  
        这是GP (Geometry Predictor)的核心评估函数，通过Sampson距离衡量匹配组合的几何质量  
          
        Args:  
            single_situation_corrs: 单一匹配组合的对应点列表  
                格式为 [[matches_of_a_pair], [], ...]  
                matches_of_a_pair: 某个区域对的对应点列表 [u0, v0, u1, v1]  
            F_from: int -- 使用哪个区域对计算F矩阵（已废弃，现在使用所有对应点）  
              
        Returns:  
            samp_dist: 平均Sampson距离，值越小表示几何一致性越好  
                     如果对应点不足则返回1e8（表示无效组合）  
                       
        Note:  
            - 该方法是GP选择最佳匹配组合的关键指标  
            - 使用对应点最多的区域对计算基础矩阵F  
            - 用该F验证所有对应点的几何一致性  
        """  
    # 检查该匹配组合是否包含任何区域对  
    if len(single_situation_corrs) == 0:  
        logger.warning(f"this situation got no matched pair")  
        return 1e8  # 返回大值表示无效组合  
          
    # 统计每个区域对的对应点数量  
    lens = []  
    for corrs in single_situation_corrs:  
        lens.append(len(corrs))  
      
    # 找到对应点最多的区域对，用于计算基础矩阵F  
    max_idx = lens.index(max(lens))  
    if lens[max_idx] < 10:   
        logger.warning("this situation got not enough corrs")  
        return 1e8  # 对应点太少，无法可靠估计F矩阵  
  
    # 步骤1: 使用对应点最多的区域对计算基础矩阵F  
    corrs_F = single_situation_corrs[max_idx]  
    F = self.calc_F(corrs_F)  # 调用RANSAC算法估计F矩阵  
  
    # 步骤2: 收集所有区域对的对应点用于一致性验证  
    corrs_other_np = []  
    for i, corrs in enumerate(single_situation_corrs):  
        # 注释掉的代码：之前可以选择跳过用于计算F的区域对  
        # if i == F_from: continue # use all corrs calc sampson dist  
        corrs_other_np += corrs  # 收集所有对应点  
  
    # 检查总对应点数量是否足够进行可靠验证  
    if len(corrs_other_np) <= self.std_match_num*1.5:  
        logger.warning("this situation got not enough corrs")  
        return 1e8  
  
    # 转换为numpy数组格式  
    corrs_other_np = np.array(corrs_other_np)  
    logger.info(f"other corrs shape is {corrs_other_np.shape}")  
      
    # 步骤3: 计算所有对应点相对于F矩阵的平均Sampson距离  
    samp_dist = self.calc_sampson(F, corrs_other_np)  
  
    return samp_dist  # 返回几何一致性度量
```

    - 选 **距离最小的组合** → 作为可信匹配
3. **输出**
    - 最可信的区域组合
    - 对每个组合区域内的点匹配（用于后续验证）

####  GR – Geometry Rejector（用几何一致性淘汰假匹配）
**目标：**

+ 解决 putative 区域对中可能存在的错误匹配。

<!-- 这是一张图片，ocr 内容为：输入GP和SAM输出的区域匹配结果 (区域对集合A) 取出一个区域对AREA_I 在该区域中收集所有点对MI 根据当前全局基础矩阵F 计算每个点的 SAMPSON 距离 YES NO 距离>    ? 该区域对几何一致; 该区域对不符合几何规则; 标记为"无效" 标记为"有效" 还有区域对? 删除所有"无效"区域对 输出过滤后的区域匹配集合 -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764491338456-92efd171-5480-4411-a78d-5e9fb41699da.png)

**流程：**

1. **输入**
    - SAM 输出的 putative area matches（初步匹配区域对）
    - 每个区域对的图像子区域
2. **核心操作**
    - 对每个区域对：
        * 在区域内运行点匹配（PM，LoFTR / SuperPoint）
        * 使用这些点拟合基础矩阵 F
        * 计算 Sampson 距离
        * 如果残差超过阈值 → **剔除该区域对**

 GR 阈值 = 所有区域自身 Sampson 残差的平均值 × 系数（α）。自残差越大，区域本身越不稳定，因此 GR 会把它剔除。 

```python
def calc_inlier_thd(self, rejecting_all_corrs, mode=0, alpha_list=[2.5]):  
    """ 计算自适应的内点阈值，用于GR几何验证  
        这是GR (Geometry Rejector)的核心组件，根据实际匹配质量动态调整过滤标准  
          
        Args:  
            rejecting_all_corrs: 所有区域对的对应点列表，格式为 [[corrs_of_pair1], [corrs_of_pair2], ...]  
                                每个corrs_of_pair是 [u0, v0, u1, v1] 格式的点对应列表  
            mode: 阈值计算模式  
                0. avg: 使用所有区域对自Sampson距离的平均值（当前唯一支持的模式）  
            alpha_list: alpha参数列表，用于生成多级阈值  
                       默认[2.5]，值越大阈值越宽松，过滤越不严格  
          
        Returns:  
            alpha_thd_dict: 多级阈值字典，格式为 {alpha: threshold}  
                           每个alpha对应一个自适应阈值，用于不同严格程度的几何验证  
          
        Note:  
            - 自适应阈值机制使系统能根据场景实际匹配质量动态调整  
            - 避免固定阈值在不同场景下的适应性问题  
            - 多级输出提供不同精度要求的过滤选项  
        """  
    # 确保alpha_list是列表类型，支持多个alpha参数  
    assert type(alpha_list) == list  
    thd = 0  
    alpha_thd_dict = {}  
      
    if mode == 0: # 平均模式：使用所有区域对的自Sampson距离平均值  
        self.self_sd_list = []  # 存储每个区域对的自Sampson距离  
        logger.info(f"calc inlier threshold by average mode")  
          
        # 步骤1: 遍历所有区域对，计算每个的自Sampson距离  
        for i, corrs in enumerate(rejecting_all_corrs):  
            # 过滤点数过少的区域对，确保F矩阵估计的可靠性  
            if len(corrs) < 100: continue  
              
            # 使用当前区域对的对应点估计基础矩阵F  
            F_temp = self.calc_F(corrs)  
              
            # 将对应点转换为numpy数组格式  
            temp_corrs_np = np.array(corrs)  
              
            # 计算当前区域对相对于自身F矩阵的平均Sampson距离  
            # 这反映了该区域对内部几何一致性的质量  
            temp_self_sd = self.calc_sampson(F_temp, temp_corrs_np)  
            self.self_sd_list.append(temp_self_sd)  
            logger.info(f"the {i} match pair calced self sampson dist = {temp_self_sd}")  
  
        # 步骤2: 基于平均自Sampson距离计算多级阈值  
        for alpha in alpha_list:  
            # 检查是否有有效的区域对用于阈值计算  
            if len(self.self_sd_list) == 0:   
                thd = 0  # 没有有效数据时设为0  
            else:  
                # 关键公式：阈值 = 平均自Sampson距离 × alpha系数  
                # alpha控制过滤的严格程度：值越大阈值越宽松  
                thd = np.array(self.self_sd_list).mean() * alpha # NOTE alpha  
            alpha_thd_dict[alpha] = thd  
  
    else:  
        # 当前只支持平均模式，其他模式待实现  
        logger.warning(f"Unspported threshold mode: {mode}")  
        raise NotImplementedError  
  
    return alpha_thd_dict
```

    - 保留几何一致性好的区域对
3. **输出**
    - 经过几何验证的可靠区域对
    - 区域内点匹配（更精准、更稳定）

| | GP（Predictor） | GR（Rejector） |
| --- | --- | --- |
| 处理对象 | **歧义的区域** | **看似可信的区域** |
| 工作方式 | 尝试多个候选组合 | 验证单个区域对 |
| 目标 | 找出最合适的组合 | 剔除不合格的区域对 |
| 几何使用 | 对组合整体算几何一致性 | 对每个区域自身算几何一致性 |
| 输出 | 生成一组新增的确定区域对 | 输出筛选后的可靠区域对 |
| 像什么 | “在多个选项里挑一个对的” | “检查挑出来的是不是错的” |




####   GMC – Global Match Collection（补点匹配，防止区域太少）
**目标：**

+ 解决区域数量太少时 F 矩阵估计不稳定的问题。

**流程：**

1. **输入**
    - 图像全图
    - 区域匹配数目不足（或者 doubtful 区域太少）
2. **核心操作**
    - 从全图采样一些额外的点匹配
    - 将这些点加入到基础矩阵 F 的拟合中
    - 保证几何约束稳定，不因为区域太少导致错误

```python
# Adaptive A2PM part ###########################################  
# 自适应区域到点匹配机制：当区域数量不足时，使用全图匹配来补充  
# if only one area; perform point matching on entire img and screen   
if len(self.rejecting_matched_area0s) <= self.filter_area_num:  
    # 记录警告：区域对数量太少，需要执行全图匹配  
    logger.warning(f"only {len((self.rejecting_matched_area0s))} area pair, perform point matching on entire img and screen")  

    # 检查是否已有全图对应点，如果没有则执行全图匹配  
    if (len(self.ori_img_corrs)==0):  
        # 执行全图匹配，返回对应点列表和numpy数组  
        self.entire_img_corrs_list, entire_img_corr_np = self._match_on_entire_img()  
    else:  
        # 使用已有的全图对应点  
        entire_img_corr_np = np.array(self.ori_img_corrs)  
        self.entire_img_corrs_list = self.ori_img_corrs  

    # 检查全图对应点数量是否足够（至少需要100个点）  
    if len(self.entire_img_corrs_list) <= 100:  
        logger.warning(f"entire img corrs num is {len(self.entire_img_corrs_list)}, too small")  
        return []  # 点数太少，返回空列表  

    # TODO for more areas!  # 未来扩展：支持多个区域的处理  

    ## 获取第一个区域对作为几何约束的参考  
    temp_area0 = self.rejecting_matched_area0s[0]  # 图像0中的参考区域  
    temp_area1 = self.rejecting_matched_area1s[0]  # 图像1中的参考区域  

    ## 使用参考区域对全图对应点进行几何筛选  
    # 基于区域的几何约束（基础矩阵F）来筛选全图中的有效对应点  
    temp_corrs = self.refine_image_corrs_by_single_area(entire_img_corr_np, temp_area0, temp_area1)  

    # 将筛选后的对应点添加到最终结果中  
    self.rejecting_all_corrs.append(temp_corrs)  

    # 返回经过几何验证的对应点列表  
    return self.rejecting_all_corrs
```

3. **输出**
    - 稳定的点匹配支持 F 拟合
    - 提升整体几何验证精度

**一句话理解：**

当区域不够时，用额外点匹配稳住几何验证。

## 3 实验
### Matching Performance（点匹配性能）
**目的：**  
验证 SGAM 是否能提升不同特征匹配器（Sparse / Semi-dense / Dense）的 **MMA（匹配精度）**。

**（1）ScanNet 结果（原文 Table I）**

 表格：**Table I – Matching performance on ScanNet**  
<!-- 这是一张图片，ocr 内容为：TABLE I VALUE RESULTS(%) OF MMA. V 63 OF MMA, WERERORT MMA VITH THREE TERESHOLDS UNDER VARIOUS MAICHNG DIFICULTIES, OUR SGAM IS AFFLED O FOUR   ASELINES, TO SHOW THE MPACT OF SEMANTIC ACCURACY TO OUR NETHOD,VE TAKEE THRENT SEMANTIC INFUTS REPORTED IN PERCENTAGE, WHICH IS IMPRESSIVE TO SHOW THE EFFECTIVENESS OF OUR METHED SCANNET:FD@5 MATTERPORT3D SCANNET:FD@10 POINT MATCHING MMA@ 3个 MMA@2个 MMA@3个 MMA@1个 MMA@3T MMA@2个 MMA@1个 MMA@11 MMA@2竹 43.32 SP+SG [29] 37.54 57.57 24.40 21.66 29.95 13.77 63.06 76.15 GT+SGAM_SP+SG 41.74+11.18% 15.94+15.80% 32.86 59.60+3.52% 26.44+8.37% 24.23+11,87% 81.46+6.98% 68.31+8.32% 44.96+3.79% 0+9.73% 25.65+5.14% 23.36+7.85% 40.82+8.73% SEEM-L[18]+SGAM_SP+SG 59.42+3.21% 31.96+6.72% 80.58:5.82% 14.9548.61% 66.684 44.42+2.54% +5.74% SEEM-T[18]+SGAM_SP+SG 25.31+3.74% 14.14+2.72% 39.34+ 31.52 22.86+5.55% 78.43+3.00% 58.65+1.87% 65.31+3.56% 43.86+1.25% +4.79% +5.25% ASPAN [10] 7.17 70.79 32.99 25.35 37.25 21.10 85.03 49.83 66.91 75.42+6.53% GT+SGAM_ASPAN 37.88 24.51+16.20% 72.81+8.83% 7.68+7.03% 28.19+11.19% 54.67+9.72% 89.40+5.14% 39.94+7.21% 14.82% 52.85+6.07% 73.76+4.19% 87.58+2.99% 23.98+13.66% 70.70+5.66% SEEM-L+SGAM_ASPAN 7.61+6.11% 27.15+7.09% 36.48 39.16+5.12% 10.58% 22.51+6.69% 38.41+3.11% 7.40+3.17% 52.11 72.84+2.89% 86.64+1.89% 35.54 SEEM-T+SGAM ASPAN 26.81+5.78% 69.44+3.79% +7.73% +4.59% SEMI-DENSE 7.44 23.97 OUADT 39] 41.72 32.79 88.31 22.67 70.40 78.46 56.92 GT+SGAMOUADT 39.43+20.25% 90.94+2.47% 45.56+9.20% 8.26+11.10% 82.32+4.939 75.96+7.89% 62.49+9.79% 26.68+17.65% 26.19+9.29% 43.05+3.17% 25.95+8.26% 73.63+4.59% 25.17+11.03% 89.30+1.12% 37.02+12.91% SEEM-L+SGAMLOUADT 7.91+6.41% 81.08+3.35% 60.55+6.38% SEEM T+SGAM OUADT 72.54+3.04% 42.88 24.50+2.23% 24.30+7.15% 59.38+4.32% 88.4040.10% 80.464.55% 36.35+10.85% 7.86+5.63% +2.76% 30.49 83.51 17.85 LOFTR [9] 9.50 67.90 22.08 36.07 65.33 46.78 GT+SGAM LOFTR 12.48+31.36% 48.31+33.93% 35.02+14.85% 19.02+6.55% 70.38+7.73% 49.10+4.95% 70.55+3.91% 88.06+5.45% 29.08+31.74% 12.27+29.16% 18.85+5.60% 27.20+23.22% 33.83+10.95% 40.25 SEEM-L+SGAM LOFTR 48.78+4.27% 87.33+4.58% 68.90+ 70.05+7.23% +1.47% +11.57% SEEM-T+SGAMULOFTR 33.17+10.55% 18.21+2.03% 11.47+20.77% 38.45 25.10+13.69% 67.98+0.12% 86.71+3.84% 47.45 +1.42% 69.52+6.40% +6.59% 32.92 COTR [35] 78.71 16.51 29.37 60.99 63.45 42.36 10.63 46.07 36.76+11.67% 12.36+16.36% 18.56+12.42% 32.64+11.16% 45.45+7.28% 64.52+5.79% 81.19+3.16% 49.82+8.15% 66.56+4.91% GT+SGAM COTR 11.73+10.40% SEEM-L+SGAMCOTR 36.54+11.00% 18.16-10.01% 63.29元 31.97+8.88% 81.04+2.96% 44.54+5.15% 66.48+4.78% 48.70+5.71% +3.76% 11.114.57% SEEM-T+SGAM COTR 36.05+9.52% 43.71+3.18% 17.60+6.60% 31.30+6.58% 80.48+2.24% 62.50+2.47% 65.89+3.85% 47.49+3.08% -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578056378-8ad18df4-caad-485d-9058-804e109e7497.png)

**实验设置**

+ 数据集：**ScanNet FD@5 / FD@10**、**MatterPort3D**
+ 指标：**MMA@1/3/5 px**
+ 匹配器：
    - 稀疏：SP+SG
    - 半稠密：ASpanFormer
    - 稠密：QuadTree / LoFTR / COTR
+ 三种语义精度对比：
    - **GT**
    - **SEEM-L（强）**
    - **SEEM-T（弱）**

**结果（Results）**

+ SGAM **在全部匹配器上都提高 MMA**
+ 即使最弱语义（SEEM-T）提升仍然明显
+ 稠密匹配器（LoFTR / QuadTree）提升比例最高

**关键结论：语义不需要特别准，只要大致区域即可显著提升点匹配精度。**

### Pose Estimation（相机位姿估计）
**（1）ScanNet Pose Estimation（原文 Table II）**

表格：**Table II – Pose estimation on ScanNet**  
<!-- 这是一张图片，ocr 内容为：TABLE I RELATTUE POSE ESTION RESULTS(G)  THE AUG OF ROSE ERROR ON SCANNET (ED ASHO)AND MATERROUTERRD WITERENT THAST  HOLD  ARERENONTED,OURSEAM IS AFFLUED ON  FASELNES TO SHOW THE NFAST OF STRACT ACT  OUR NETIOD, TANTES TUREEDUNERRT SRNANTE NRUTS, SGAM USING SROUND RUTE SSAM USNG SEENTE SEENTE SSAMUSNG SEENFT  TY IMPROVEMENT ACHIEVED BY SGAM IS ALSO REPORTED IN PERCENTAGE. SCANNET FD@10 SCANNET:FD@5 MATTERPORT3D POSE ESTIMATION AUC@209个 AUC@109 AUC@59 AUC@301 AUC@201 AUC@59 AUC@10个 AUC@20 AUC@101 SP+SG[29] 64.47 76.46 73.58 53.11 29.54 37.61 67.46 16.39 86.61 GT+SGAM_SP+SG 78.61+2.81% 75.98+3.26% 69.20+2.58% 39.31+4.52% 17.93+9.40% 55.87+5.20% 31.72+7.38% 66.87+3.72% 88.72+2.44% SPARS 75.03+1.97% 77.60+1.49% 53.91+1.51% 68.61+1.70% SEEM-L[18+SGAM_SP+SG 87.44+0.96% 17.15+4.64% 38.51+2.39% 65.46+1.54% 31.53+6.74% 77.35+1.16% SEEM-T [18]+SGAM_SP+SG 74.62+1.41% 68.33+1.30% 16.85+2.81% 87.13+0.60% 37.95 +0.90% 53.26+0.28% 30.37+2.81% 65.11+0.99% 70.73 77.41 27.81 79.84 70.42 18.35 ASPAN[10] 58.51 43.98 80.19 74.24. 73.60+4.06% 81.52+5.30% 20.50. GT+SGAM_ASPAN 84.53+5.87% 30.08+8.17% 48.49+10.26% 60.78+3.89% 85.83+7.02% +5.43% +11.71% 72.32+2.24% 73.02+3.69% 83.40+4.46% 80.59+4.10% 59.17+ 45.52+3.51% SEEM-L+SGAM_ASPAN 85.20+6.24% 29.38+5.66% 19.74+7.56% +1.13% 72.02+2.27% 82.54+3.38% 79.94+3.26% 83.32+3.90% 28.87+3.81% 58.87+0.62% 71.84+1.56% 44.34+0.83% 18.641.56% SEEM-T+SGAM ASPAN 69.77 79.39 39.96 74.25 16.53 74.96 69.48 59.27 26.98 QUADT [39] 72.55 71.83+2.95% 42.21+5.63% GT+SGAM_QUADT 61.82+4.31% 28.11+4.19% 76.82+2.49% 75.89+2.27% 82.10+3.41% 18.90+14.33% +4.41% 71.92+3.51% 71.96+3.14% SEEM-L+SGAM_QUADT 27.25+1.00% 75.60+1.81% 76.52+2.08% 61.78+4.23% 41.564.00% 17.83+7.82% 81.4342.56% SEEM-T+SGAM.OUADT 71.77+2.86% 27.10+0.44% 71.47+2.86% 75.80+1.13% 75.06+1.09% 40.22+0.65% 60.97+2.87% 16.841.88% 80.661.59% LOFTR 9 74.29 27.79 17.98 78.45 78.99 67.69 58.71 38.19 69.81 71.22+ 73.12+4.74% 83.13+5.24% 32.28+16.16% GT+SGAM LOFTR 45.37+ 18.14+0.89% 79.31+6.76% 59.50+1.35% 84.44+7.64% +5.21% +18.80% 83.93+6.99% 70.25+3.79% 71.02+1.74% SEEM-L+SGAM_LOFTR 80.91+2.42% 30.86+11.04% 60.37+2.83% 42.13+10.32% 18.04+0.31% 79.23+6.65% 40.6346.39% SEEM-T+SGAM LOFTR 69.53+2.72% 79.78+1.01% 17.47-2.85% 82.32 78.03+5.03% 59.01+0.50% 70.58+1.10% 30.00+7.94% +4.93% COTR [35] 72.55 74.11 17.80 25.08 51.92 63.36 78.48 34.08 66.91 DENSE 79.22, GT+SGAM COTR 71.18+6.38% 84.22+7.31% 53.99 80.17+10.50% 28.31+12.88% 41.25+21.04% 68.29+7.78% 19.20+7.87% +6.90% +3.99% SEEM-L+SGAM COTR 78.78+8.59% 26.99+7.62% 69.67+4.12% 18.12+1.82% 36.69+7.67% 52.80+1.70% 83.84+6.83% 78.15+5.46% 66.16+4.42% 77.98+5.32% 17.93+0.71% 25.79元 35.79+5.01% 78.43+8.10% 52.07+0.28% SEEM-T+SGAMCOTR 69.40+3.72% 65.91+4.03% 83.07+5.85% +2.84% -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578099189-33bd2129-01bd-4dd3-adac-be1d881cddf0.png)

**实验设置**

+ 输入：点对应（来自不同匹配器 + SGAM）
+ 方法：RANSAC + Essential Matrix
+ 数据集：ScanNet FD@5 / FD@10
+ 指标：AUC@5° / 10° / 20° （判断姿态误差（Rotation/Translation）是否小于 5°、10°、20°。   AUC 越大 → 位姿越准。 ）

**结果**

+ SGAM 使 **所有匹配器的 AUC 都显著增加**
+ 特别是在 FD@10（更难场景），提升更大

**（2）KITTI360（室外） – Table III**

表格：**Table III – KITTI360 Pose Estimation**

<!-- 这是一张图片，ocr 内容为：TABLE II  RETATIVEROSE ESTINATION RESUUTS (R) ON SITTI3EB DAFASET, WE CONFARE TWO DIFFERENANTIC FOR OUR ANETHO SGAM USING GROUND TRUTH (GT) AND SGAM USING SEEM-L - SQE.00 SEG.03 SEQ.05 POSE ESTIMATION AUC@10个 AUC@20个 AUC@59个 AUC@10个 AUC@59个 AUC@ 30个 AUC@20个 AUC@20个 AUC@10 SFD2 16] 72.71 83.24 94.31 88.04 90.95 80.52 92.30 63.18 68.58 SPARSE SP[7+SG[29] 91.34 74.24 93.91 89.15 81.34 85.44 96.03 63.91 69.65 GT+SGAM SP+SG 82.78+1.77% 72.17+3.62% 94.37+0.49% 92.61+1.39% 90.10+1.07% 75.36+1.51% 97.26+1.28% 65.01+1.72% 86.31+1.02% SEEM-L [18]+SGAM_SP+SG 92.17+0.91% 75.23+1.33% 94.11+0.21% 71.93+3.27% 82.53+1.46% 86.06+0.73% 96.99+1.00% 64.85+1,47% 89.87+0.81% 77.64 75.57 87.16 91.20 57.38 61.19 87.95 83.00 ASPAN[10] 68.01 GT+SGAM ASPAN 81.86+5.44% 89.78+3.00% 92.96+1.94% 63.31+10.35% 80.22+6.15% 73.81+8.53% 90.63+3.05% 66.05+7.95% 86.43+4.14% SEMI-DENSE SEEML 18]+SGAM_ASPAN 89.70. 92.95+1.92% 90.38+2.76% 73.76+8.46% 81.67+5.19% 63.06+9.91% 66.18+8.17% 86.40+4.10% 80.07+5.95% +2.92% 77.77 59.93 77.24 90.39 QUADT [39] 88.40 81.81 58.93 66.47 88.27 67.77+13.10% 92.68+2.53% 73.40+10.42% 82.68+6.32% 81.97+6.11% 90.83 GT+SGAM OUADT 91.04+3.13% 86.0545.18% 65.98+11.96% -2.75% 90.67+2.57% 73.33 81.77+5.86% SEEM-L [18]+SGAM OUADT 92.65+2.51% 82.63+6.25% 67.72+13.01% 91.01+3.10% 86.00+5.12% 65.92+11.86% +10.32% 80.19 LOFTR 9] 65.11 84.87 92.11 89.95 71.53 80.98 63.54 90.10 86.95+2.45% 70.55+8.36% GT+SGAM LOFTR 84.79+4.71% 92.33+2.48% 92.20+2.50% 75.40+5.41% 84.44+5.30% 93.03+1.00% 69.68+9.66% 70.20+7.83% SEEM-L (18]+SGAM LOFTR 87.27 84.54+4.39% 92.16+2.29% 92.01 93.18+1.16% 76.21 84.114.89% 69.51+9.39% +2.83% +2.29% +6.54% 77.61 80.92 COTR [35] 89.22 62.76 79.36 88.55 86.67 58.69 66.97 85.89+6.15% GT+SGAMLCOTR 91.32+2.35% 81.70+5.27% 72.40+8.11% 90.96+2.72% 66.17+12.75% 88.30+1.89% 67.55+7.63% 82.01+3.34% 90.17+1.06% SEEM-L [18]+SGAM_COTR 70.17+4.77% 90.18+1.85% 80.90+4.23% 84.21+4.07% 88.37+1.96% 64.39+9.72% 81.1942.31% 66.69+6.26% -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578141563-1e6ae89c-9438-4f7a-bb2f-674c506df763.png)

**结果**

+ 在室外驾驶场景中 SGAM 同样提升 AUC
+ 室外语义简单 → 提升幅度略小  
但仍一致提升，说明 SGAM 具有跨场景稳定性。

**（3）YFCC100M（互联网照片） – Table V**

表格：**Table V – YFCC100M Pose Estimation**  
<!-- 这是一张图片，ocr 内容为：TABLE V RELATIVE POSE ESTIMATION RESULTS(%)ON YFCC100M.TWO DIFFERENT SEMANTIC INPUTS FOR OUR METHOD ARE COMPARED: SGAM USING SEEM-L AND SGAM USING SEEM-T . YFCCI00M POSE ESTIMATION AUC@100个 AUC@59个 AUC@209个 39.25 PATS [15] 76.38 60.77 26.82 62.17 SP+OANET[31] 45.04 28.45 67.19 SP+SG[29] SPARSE 48.60 70.02+4.21% OETR 141+SP+SG 31.51+10.76% 50.61+4.14% 29.541 69.64+3.65% SEEM-LI8+SGAM SP+SG 50.48+3.87% +3.83% 68.26+1.59% SEEM-T[18]+SGAM_SP+SG 29.14+2.43% 50.01 +2.90% 75.54 38.96 ASPAN[10] 59.35 76.22+0.90% 39.31+0.90% OETR+ASPAN 60.13+1.31% 76.34+1.06% 39.90元 SEEM-L+SGAM_ASPAN 60.36+1.70% +2.41% 76.21+0.89% 39.77 60.24+1.50% SEEM-T+SGAM_ASPAN 1+2.08% SEMI-DENSE 76.57 40.73 QUADT [39] 61.19 77.08+0.67% 41.46+1.79% OETR+OUADT 62.15+1.57% SEEM-L+SGAMOUADT 76.79+0.29% 41.32 61.33+0.23% +1.45% 77.02+0.58% SEEM-T+SGAM OUADT 41.07- 61.44+0.41% +0.83% 41.12 77.01 LOFTR [9] 61.43 41.83 77.35+0.44% OETR+LOFTR 62.16+1.19% -1.73% 41.54+0.95% SEEM-L+SGAM LOFTR 77.12+0.14% 61.72+0.47% 41.33. 77.08 SEEM-T+SGAM LOFTR 61.67+0.39% +0.09% -0.51% 43.12 79.13 DKM[12] 63.78 DENSE 43.28+0.37% OETR+DKM 64.27+0.77% 79.34+0.27% 43.77+1.51% 79.94+1.01% SEEM-L+SGAM DKM 64.12+0.53% SEEM-T+SGAM DKM 79.77+0.81% 43.56+1.02% 63.99+0.33% -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578167006-3c25af5c-cee3-4d8c-b890-da3158620c1c.png)

**结果**

+ 即使语义最差（SEEM-T）仍能提升
+ 互联网图片变化巨大 → SGAM 依然有效

**说明 SGAM 是强鲁棒性的几何增强框架。**

### Area Matching（区域匹配性能）
 表格：**Table VI – Area matching performance on ScanNet**  
<!-- 这是一张图片，ocr 内容为：TABLE IV RELATIVE POSE ESTIMATION RESULTS (%) ON SCANNET1500 OUR METHOD BENCHMARK. OBTAINS THE SEMANTIC PRIOR BY SEEM-L. THE BEST AND SECOND RESULTS ARE HIGHLIGHTED. SCANNETL500 BENCHMARK POSE ESTIMATION AUC@59个 AUC@10个 AUC@201 64.30 26.00 46.90 PATS [15] 26.90 SP[7]+OANET[3]] 11.80 43.90 16.20 33.80 SP+SG [29] 51.80 MKPC 13]+SP+SG 34.11+0.92% 16.18-0.12% 52.47+1.29% SEEM-L [18]+SGAM_SP+SG 34.77+2.87% 52.13 17.33 +0.64% 6.98% 25.78 46.14 ASPAN [IO] 63.32 SEMI-DENSE 27.51+6.71% 48.01+4.05% 65.26+3.06% SEEM-L+SGAM_ASPAN 25.21 44.85 QUADT [39] 61.70 63.40+2.76% SEEM-L+SGAM OUADT 25.53 46.02+2.60% -1.27% LOFTR [9] 22.13 40.86 57.65 23.39+5.69% 41.79+2.28% SEEM-L+SGAM LOFTR 58.74+ +1.89% 29.40 68.31 50.74 DKM[12] 69.31+1.48% 52.34+3.10% 30.61+4.12% SEEM-L+SGAMLDKM -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578188542-a0c72f61-d045-441a-8797-897050890b7d.png)

**目的**

验证 SAM 和 SGAM 能否正确找到跨图像的语义区域对（Area-to-Area）。

**方法**

+ 继续用 ASpanFormer / LoFTR / COTR
+ 指标：
    - **AOR**（Area Overlap Ratio） （AOR 越大 → 匹配越准 ）
    - **AMP**（Area Matching Precision）（ AMP 越高 → 说明 SGAM 匹配到了更多正确区域，误匹配更少。）

**结果**

+ SGAM 在所有难度（FD@5/10/30）中均显著优于 SAM
+ 越难（FD 越大），提升越大
+ SGAM 的区域匹配更精确、更稳定

**为 GP/GR/GMC 的后续几何验证打下基础。**

### Semantic Region Behavior（SOA / SIA 行为分析）
表格：**Table VII – SOA/SIA Matching Analysis**  
<!-- 这是一张图片，ocr 内容为：TABLE VII AREA MATCHING PERFORMANCE OF TWO SAM AREAS AND GP. WE CONSTRUCT AREA MATCHING EXPERIMENTS ON SCANNET FOR MATCHING OF TWO SEMANTIC AREAS. ALONG WITH GP INTEGRATED WITH FOUR POINT MATCHERS. THE EFFECT OF TWO DIFFERENT SEMANTIC INPUTS IS ALSO EVALUATED. AOR AND AMP (WITH THRESHOLD T - O.7) UNDER DIFFERENT MATCHING DIFFICULTIES (EACH WITH 1500 IMAGE PAIRS) ARE REPORTED ALONG WITH THE AREA NUMBER PER IMAGE (NUM). THE BEST AND SECOND ROOM ND RESULTS UNDER EACH SEMANTIC INPUT SETTING AND FD SETTING ARE HIGHLIGHTED. FD@5 FD@30 FD@10 METHOD AMPT NUM AORT AMPT NUM NUM AMPT AORT AORT 2.30 70.84 3.13 94.10 85.94 2.91 SOA MATCH 68.36 85.26 91.76 83.67 84.35 91.91 62.17 2.38 83.50 SIA MATCH 66.94 2.01 1.26 GT SEM. GP ASPAN 86.97 96.70 81.26 84.83 86.59 89.59 82.37 88.47 8786 GP OUADT 96.82 87.91 84.98 0.36 0.26 0.50 8751 GP LOFTR 92.18 73.81 87.42 95.73 86.48 89.37 GP COTR 95.27 73.12 82.59 8646 86.58 3.35 72.25 4.94 2.62 81.22 62.74 86.33 SOAMATCH 81.14 89.94 SEEM-L SEM. 83.39 72.53 2.21 51.01 1.76 65.85 83.46 77.19 SIA MATCH 2.51 8490 84.51 83.34 GPASPAN 75.43 63.02 90.66 85.03 74.03 GP OUADT 87.26 63.16 81.54 82.06 0.64 1.61 0.57 GP LOFTR 87.23 74.25 89.94 64.44 82.59 83.91 73.87 85.39 80.73 63.48 89.65 81.54 GP COTR -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578545179-6fd43b41-216a-4e72-a1f9-ef25edb03831.png)

**目的**

分析两类语义区域：

+ **SOA（语义对象区域）**：大而稳定
+ **SIA（交叉区域）**：数量多但不稳定

**结果**

+ SOA 数量在 GT / SEEM-L / SEEM-T 下都非常稳定
+ SIA 数量更多但更难匹配
+ SOA 匹配精度始终高于 SIA

说明：  
**语义区域（SOA/SIA）本身就是结构化的，非常适合做区域匹配。**

### Ablation Study（消融实验）
表格：**Table X – Ablation on Different Components**  
<!-- 这是一张图片，ocr 内容为：TABLEX Y OF COMPONENTS. WE CONDUCT THE DECOMPOSING ABLATION STUDY O COMPONENT EXPERIMENT OF SGAM ASPAN ON SCANNET WITH FD@ 15, USING IK IMAGE PAIRS. THE AREA MATCHING AND POSE ESTIMATION PERFORMANCE ARE REPORTED. THE NUMBERS OF AREA MATCHES (NUM) ARE ALSO REPORTED. AMP个 NUM AUC@5 0个 AUC@10 AOR SOA SIA GP GR GMC AUC@20 43.77 32.96 79.39 78.88   2.74 53.80 694211 74.64 31.72 40.66 46.89 77.41 48.26 52.94 34.02 74.95 453 52.97 39.54 79.03 5.57 64.54 76.30 62.01 49.24 3.73 79.01 73.32 78.00 79.18 62.58 49.73 78.27 73.17 4.01 63.38 79.18 50.50 78.27 4.01 74.75 -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578310296-dab1a88a-f272-4276-919b-160ccb0ec4e6.png)

**目的**

验证三个核心模块是否必要：

+ **GP（Geometry Prior）**
+ **GR（Geometry Rejector）**
+ **GMC（Global Multi-layer Consistency）**

**实验设计**

每次去掉一个模块，测：

+ AOR / AMP（区域）
+ Pose AUC（几何）

**结果**

+ 去掉 GP → 区域误匹配大增
+ 去掉 GR → 几何一致性崩溃
+ 去掉 GMC → pose AUC 快速下降
+ 全组件 SGAM → 最佳性能

 **三个模块缺一不可，互相补充。**

### Runtime Analysis（运行时间分析）
表格：**Table XII – Running Cost of SGAM Modules**

<!-- 这是一张图片，ocr 内容为：TABLE XII TIME CONSUMPTION COMPARISON. THE EXPERIMENT IS CONDUCTED ON SCANNET WITH FD@10. THE TIME CONSUMPTION OF EACH COMPONENT OF OUR METHOD WITH SPECIFIC INPUT SIZE IS REPORTED. DIFFERENT TIME CONSUMPTION COMES FROM DIFFERENT BASELINES COUPLED WITH OUR METHOD ARE INVESTIGATED AS WELL. THE TIME OF BASELINES ARE ALSO REPORTED. INPUT SIZE TIME/S 640X480 256X256 640X480 PM2 GP GR PMER SGAM GMC SAM 0.88 0.19 0.021 0.042 ASPAN 0.20 QUADT 0.85 0.023 0.18 0.17 0.040 0.62 LOFTR 0.19 0.041 0.86 0.18 0.018 COTR 23.85 29.14 2.13 2.54 56.04 POINT MATCHER INCORPORATED BY SGAM; POINT MATCHING ON THE ENTIRE IMAGES; -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1764578336550-edd3ecac-3275-4285-96e0-9fae2537810d.png)

**目的**

分析 SGAM 是否造成额外时间消耗。

**结果**

+ GR 最耗时（因为区域内部多次 PM 匹配）
+ 对 transformer-based 匹配器（COTR/LoFTR）  
**SGAM（裁剪成256×256）反而比原始640×480更快**
+ 总体时间可接受，且可并行优化

SGAM **轻量级 + 可加速**，不是一个重模型。

### SGAM实验总结
**SGAM 实验证明：**

+ **点匹配更准**（Table I）
+ **位姿估计更稳**（Table II–III–V）
+ **区域匹配更精准**（Table VI）
+ **SOA/SIA 区域具有强稳定性**（Table VII）
+ **三个模块 GP/GR/GMC 都必不可少**（Table X）
+ **速度可接受，甚至有优化空间**（Table XII）
+ **在大视角、重复纹理、光照差等困难场景提升最大**

