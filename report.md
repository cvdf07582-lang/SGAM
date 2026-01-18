[main.pdf](https://leedong25.yuque.com/attachments/yuque/0/2025/pdf/60748161/1765785332096-2da6407c-9d63-49a1-bede-02946189e6c4.pdf)

论文链接[https://link.springer.com/chapter/10.1007/978-3-031-43907-0_56](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_56)

github[https://github.com/LeonBP/SuperPointTrackingAdaptation](https://github.com/LeonBP/SuperPointTrackingAdaptation)

## 1 论文总结
**研究背景与问题动机**  
内窥镜检查在临床医学中具有核心地位，是消化道疾病筛查、早期癌变发现以及微创治疗的重要技术基础。随着成像设备的发展，内窥镜检查能够产生大量连续视频数据，使得利用计算机视觉技术进行三维重建成为可能。

**为什么内窥镜 3D 重建困难**  
尽管从内窥镜视频中恢复三维结构具有重要应用价值，但现有基于 Structure-from-Motion（SfM）的方法在该场景下面临根本性困难：组织表面纹理极少、图像中存在大量镜面反射伪影，以及场景整体呈现非刚性形变特性。这些因素共同导致特征点难以稳定检测和匹配，使三维重建在真实结肠镜数据中尤为困难。

已有研究表明，在理想条件下可以从短视频片段中恢复小规模三维模型，但其适用范围非常有限。其核心瓶颈在于缺乏数量充足且在时间上可重复的高质量特征点。

**研究切入点**  
基于上述分析，论文将研究重点放在特征点学习上。虽然 SuperPoint 在自然图像中表现优异，但其基于 Homographic Adaptation 的训练假设并不适用于内窥镜场景。因此，本文提出利用真实内窥镜视频中成功完成三维重建的特征点作为监督信号，并据此提出 Tracking Adaptation 方法，引出后续内容。

## 2 相关工作
**内窥镜三维重建研究现状**  
内窥镜和腹腔镜场景下的三维重建是一个长期存在但尚未解决的问题。已有工作尝试将传统 SLAM 与 SfM 方法应用于内窥镜视频，并构建了相关数据集。这些研究验证了在受限条件下重建的可行性，但同时表明，在真实临床数据中，重建结果往往规模小、稀疏且不稳定。

**针对特殊挑战的改进尝试及其局限**  
部分研究尝试针对内窥镜场景中的非刚性形变和强反光问题进行建模或抑制。这类方法虽然体现了对内窥镜特殊性的认识，但通常系统复杂、假设较强，且并未从根本上解决特征点不稳定的问题。

**特征点方法仍是瓶颈**  
论文进一步指出，无论是传统的人工特征（如 SIFT、ORB），还是近年来提出的学习型特征（如 SuperPoint、R2D2、D2-Net），在自然场景中表现良好，但直接迁移到内窥镜场景时仍难以满足 SfM 对稳定性和几何一致性的要求，其根本原因在于训练数据和训练目标与内窥镜三维重建任务不匹配。

**本文定位**  
因此，本文并不重新设计匹配网络，而是关注如何学习更适合 SfM 的特征点。作者提出利用真实内窥镜视频中成功重建的特征点作为监督信号，引导网络学习真正有利于三维重建的特征，这一思路直接引出 Tracking Adaptation 方法。

## 3 论文方法
### 3.1 方法核心思想
作者提出的 Tracking Adaptation 并不是通过设计新的网络结构或引入额外模块来提升性能，而是从**监督方式**入手，重新定义什么样的特征点才是“对三维重建有用的特征点”。其核心思想可以概括为一句话：**用真实内窥镜视频中“已经被证明可以成功参与三维重建的特征点”，来反过来监督特征点网络的学习过程。**

**为什么原始 SuperPoint 的训练策略不适用**  
作者首先分析了 SuperPoint 中使用的** Homographic Adaptation**。该策略通过对单张图像施加随机单应变换，要求网络在这些变换下保持特征点检测与描述的一致性。这一做法隐含了两个关键假设：**一是图像中的局部区域可以近似看作平面，二是不同视角之间的变化可以用刚性的几何变换来描述。**

作者指出，这两个假设在内窥镜场景中普遍不成立。内窥镜图像所观测到的组织表面通常是弯曲的，并且会随着时间产生明显的非刚性形变，同时还伴随强烈的镜面反射。因此，继续采用 Homographic Adaptation 作为监督信号，会迫使网络学习一种与真实几何关系不匹配的一致性，从而限制其在内窥镜三维重建任务中的效果。

<!-- 这是一张图片，ocr 内容为：HOMOGRAPHIC ADAPTATION UNWARP WARP GET POINT APPLY SAMPLE RANDOM UNLABELED LMAGE HEATMAPS DETECTOR LMAGES RESPONSE HOMOGRAPHY A INTEREST POINT HI AGGREGATE SUPERSET HEATMAP DODD H2 BASE DETECTOR HN FIRURE S, HOMORRPHIC ADAPTATION. HOOSTIC  ADIPERPATION IS A FORM OF SELFSAPTAN INTEREST POIN DETECTOR RRAINED WITH CONVOLURD NEURD NEIVORKS,THE EQUATIRE IS MATHEMATICALY IN EQUATIO -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766728199443-ddfcb311-1dbc-4e36-bebb-f12989c93825.png)

 ① 输入：Unlabeled Image A（无标注图像）  

 ② Sample Random Homography（采样随机单应变换 H₁…Hₙ）  **Homography 是描述“同一个平面在两个视角下如何对应”的变换。**数学上是一个 **3×3 矩阵**，可以表示：视角变化，平移，旋转，缩放，透视变化。使用这个矩阵包含隐含假设：**图像内容可以近似看成“一个平面”  **

 ③ Warp Images（对图像做几何变换）  把原图 A用 H₁、H₂、…、Hₙ变成 N张**几何扭曲后的图像 ** 

 ④ Apply Detector（用同一个 detector 检测关键点）  对每一张 warped image用 **同一个基础特征点检测网络**输出关键点响应图（heatmap）

 ⑤ Get Point Response（得到关键点响应）  

 ⑥ Unwarp Heatmaps（把响应图变回原图坐标）  把每个响应图用 **H⁻¹**变回到原始图像 A 的坐标系， 这样就可以在**同一个坐标系下比较结果**

 ⑦ Aggregate Heatmap（聚合多个响应）   叠加 / 求平均  

 ⑧ 输出：Interest Point Superset   一张增强版的关键点热图 ，被认为是：**几何一致，稳定。可重复**的特征点集合，这些点，被当作 **“伪标签”**用来监督 SuperPoint 的训练，后续转成 cell 级 65 维伪标签（裁剪热力图的一个 8×8 区域，找：最大响应像素位置，判断：最大值是否超过阈值，生成：65 维 one-hot 标签）

<!-- 这是一张图片，ocr 内容为：(A)LNTEREST POINT PRE-TRAINING (B)LNTEREST POINT SELF-LABELING (C)JOINT TRAINING SUPERPOINT LABELED INTEREST INTEREST UNLABELED IMAGE PSEUDO-GROUND POINT LMAGES POINT LOSS TRUTH INTEREST BASE DETECTOR POINTS DESCRIPTOR HOMOGRAPHIC WARP TRAIN LOSS BASE DETECTOR ADAPTATION JOOD INTEREST POINT LOSS [SEE SECTION 3] [SEE SECTION 4] [SEE SECTION 5] FSURE 2 SELT SUPERNSED  TRALNING OVERVERTEREST OUR SEFSUPERV'EED APPROACH, WE (A) INTEREST POIN   ON SYUHETE DULS AND (D) APPY A DOVEL HOMASET UNLEPATON PROCEDURE IO AUTONATERATER INASES STOM  UNLED DON THE  BENERICD ATEES ASED IO  REANA  LULY COM AN AN  NEIVONARK MEIV  EXTAT  EXT  EXT POMD  DESEN AN  A -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766914624227-f69ba629-e5a3-4618-a4ac-fdf24860a6b5.png)

(a) Interest Point Pre-Training 合成数据预训练检测器

(b) Interest Point Self-Labeling，Homographic Adaptation 生成伪标签

(c) Joint Training 同时训练检测头 + 描述子头 输出像素级的特征点响应图，用来选出稳定的关键点位置；和对应的高维特征描述子，用于跨视角的特征匹配。

<!-- 这是一张图片，ocr 内容为：INTEREST POINT DECODER CONV W/8 INPUT SOFTMAX RESHAPE ENCODER H/8 W TODAN 65 DESCRIPTOR DECODER CONV W/8 L2 BI-CUBIC INTERPO LATE NORM H/8 H D D FIGURE 3. SUPERPOINT DECODERS. BOTH DECODERS OPERATE ON A SHARED AND SPATIALLY LY REDUCED REPRESENTATION OF THE INPUT. TO KEEP THE MODEL FAST AND EASY TO TRAIN, BOTH DECODERS THE USE NON-LEARNED HXW UPSAMPLING TO BRING THE REPRESENTATION BACK TO IR HE -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766920446132-4dc86187-7007-46fe-bcae-eb13b79b2606.png)  

因为backbone 是一个 CNN , 做 3 次 stride=2 的下采样, 特征图上的一个像素 = 原图的 8×8,因此每个 cell有64+1个通道

 检测损失 =用 Homographic Adaptation 生成的“稳定点”去监督网络预测这些点的位置 ， 检测头采用 65 维 softmax 与交叉熵损失，在每个 8×8 cell 内进行位置分类。Homographic Adaptation 提供的伪标签指定 cell 内最稳定的像素位置或 dustbin，交叉熵通过局部竞争机制引导网络将概率集中于稳定点，从而实现无人工标注的特征点检测学习。 

 描述子损失 =利用已知的 homography，自动构造点的对应关系来监督学习 ，即用 H 把 某个点 p 映射到 p′如果：p′ 和某个 cell 中心距离很近，则为这是正样本

```python
def forward(self, data):  
    """   
    前向传播，联合计算未处理的点和描述子张量  
    输入  
        x: 图像pytorch张量，形状 N x 1 x H x W  
    输出  
        semi: 输出点张量，形状 N x 65 x H/8 x W/8  
        desc: 输出描述子张量，形状 N x 256 x H/8 x W/8  
    """  
  
    # ==================== 共享编码器 ====================  
    # 第一个卷积块：64通道，包含卷积、BN、ReLU和池化  
    x = self.relu(self.bn1a(self.conv1a(data['image'])))  
    conv1 = self.relu(self.bn1b(self.conv1b(x)))  
    x, ind1 = self.pool(conv1)  # 最大池化，返回索引用于后续可能的上采样  
      
    # 第二个卷积块：64通道  
    x = self.relu(self.bn2a(self.conv2a(x)))  
    conv2 = self.relu(self.bn2b(self.conv2b(x)))  
    x, ind2 = self.pool(conv2)  
      
    # 第三个卷积块：128通道  
    x = self.relu(self.bn3a(self.conv3a(x)))  
    conv3 = self.relu(self.bn3b(self.conv3b(x)))  
    x, ind3 = self.pool(conv3)  
      
    # 第四个卷积块：128通道（无池化）  
    x = self.relu(self.bn4a(self.conv4a(x)))  
    x = self.relu(self.bn4b(self.conv4b(x)))  
      
    # ==================== 检测头 ====================  
    # 检测头处理：生成65通道的热力图（8x8网格 + dustbin通道）  
    cPa = self.relu(self.bnPa(self.convPa(x)))  
    semi = self.bnPb(self.convPb(cPa)) # 这是原始输出，形状 N x 65 x H/8 x W/8  
      
    # ==================== 关键点后处理 ====================  
    # 复制semi张量进行后处理  
    scores = torch.clone(semi)  
    # Softmax激活，移除dustbin通道（第65通道）  
    scores = torch.nn.functional.softmax(scores, 1)[:, :-1]  
    b, _, h, w = scores.shape  
      
    # 将65通道重塑为8x8网格，实现亚像素精度  
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)  
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)  
      
    # 非极大值抑制，去除邻近的重复检测  
    scores = simple_nms(scores, self.config['nms_radius'])  
      
    # 提取关键点：找到超过阈值的点  
    keypoints = [  
        torch.nonzero(s > self.config['keypoint_threshold'])  
        for s in scores]  
      
    # 获取对应的关键点置信度分数  
    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]  
      
    # 移除图像边界附近的关键点  
    keypoints, scores = list(zip(*[  
        remove_borders(k, s, self.config['remove_borders'], h*8, w*8, self.bordermask)  
        for k, s in zip(keypoints, scores)]))  
      
    # 保留置信度最高的K个关键点  
    if self.config['max_keypoints'] >= 0:  
        keypoints, scores = list(zip(*[  
            top_k_keypoints(k, s, self.config['max_keypoints'])  
            for k, s in zip(keypoints, scores)]))  
      
    # 坐标转换：从(h,w)格式转换为(x,y)格式  
    keypoints = [torch.flip(k, [1]).float() for k in keypoints]  
      
    # ==================== 描述子头 ====================  
    # 描述子头处理：生成256维描述子  
    cDa = self.relu(self.bnDa(self.convDa(x)))  
    desc = self.bnDb(self.convDb(cDa))  
      
    # L2归一化描述子  
    dn = torch.norm(desc, p=2, dim=1)  
    descnorm = desc.div(torch.unsqueeze(dn, 1))  
      
    # ==================== 描述子采样 ====================  
    # 对描述子进行额外的L2归一化  
    descriptors = torch.nn.functional.normalize(desc, p=2, dim=1)  
      
    # 在关键点位置采样描述子（双线性插值）  
    descriptors = [sample_descriptors(k[None], d[None], 8)[0]  
                   for k, d in zip(keypoints, descriptors)]  
      
    # 返回包含所有结果的字典  
    return {  
        'keypoints': keypoints,      # 检测到的关键点列表  
        'scores': scores,            # 关键点置信度分数列表  
        'descriptors': descriptors,   # 关键点描述子列表  
        'semi': semi,               # 原始检测头输出  
        'desc': descnorm            # 归一化的稠密描述子  
    }
```

### 3.2 基于真实 SfM 结果的监督信号构建
基于上述问题，作者提出直接利用真实内窥镜视频的 SfM 重建结果作为监督来源。具体做法是，从 **EndoMapper 数据集**中选取长度约为** 4–7 秒的视频片段**，并使用 COLMAP 对这些片段进行三维重建。为了提高监督信号的可靠性，作者分别采用了传统 SIFT 特征和基于 SuperPoint + SuperGlue 的特征配置进行重建，从而获得一组在真实数据上能够被稳定三角化的三维点。**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">SIFT</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">作为传统特征提取方法，具有成熟的几何验证机制；</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">SuperPoint+SuperGlue</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">作为深度学习方法，在复杂场景下可能表现更好。两种方法的交集确保了特征在不同算法下的稳定性</font>

作者强调，**这些三维点本身已经通过多视几何验证，说明它们对应的图像特征在空间和时间上具有较高稳定性，因此非常适合作为“高质量特征点”的监督信号。**

```python
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()  
    parser = argparse.ArgumentParser(description='Reconstruct with SuperPoint feature matches.')  
      
    # 必需参数：视频ID、簇ID、模型名称  
    parser.add_argument('video', type=str, default='00033')  # 视频序列标识符  
    parser.add_argument('cluster', type=str, default='19')   # 数据簇标识符或'None'  
    parser.add_argument('model', type=str, default='sp_bf')  # 输出目录的模型名称  
      
    # 特征检测参数  
    parser.add_argument('--featuretype', type=str, default='superpoint')  # 特征检测器类型  
    parser.add_argument('--superpoint', type=str, default='superpoint_ucluzlabel100_specga_9-4_d3/checkpoints/superPointNet_200000_checkpoint.pth.tar')  # SuperPoint权重路径  
    parser.add_argument('--keypoint_threshold', type=str, default='0.015')  # 关键点检测阈值  
      
    # 特征匹配参数  
    parser.add_argument('--matchtype', type=str, default='bruteforce')  # 匹配算法类型  
    parser.add_argument('--superglue', type=str, default='superglue_pretrained/superglue_indoor.pth')  # SuperGlue权重路径  
    parser.add_argument('--maxdistance', type=str, default='1.0')  # SIFT匹配的最大描述符距离  
      
    # COLMAP重建参数  
    parser.add_argument('--reperror', type=str, default='4')      # 最大重投影误差（像素）  
    parser.add_argument('--overlap', type=str, default='10')      # 序列匹配的重叠窗口大小  
    parser.add_argument('--minsize', type=str, default='50')      # 有效重建的最小图像数量  
      
    args = parser.parse_args()  
  
    # ==================== 参数处理和模型名称构建 ====================  
    video = args.video  
    # 如果cluster为'None'，使用完整视频模式（Full_前缀）  
    cluster = args.cluster if args.cluster != 'None' else 'Full_'+args.overlap  
    model = args.model  
      
    # 根据特殊参数修改模型名称，用于实验追踪  
    if args.maxdistance != '1.0':  
        model = model + '_md' + args.maxdistance.replace('.', '')  # 添加最大距离标记  
    if args.keypoint_threshold != '0.015':  
        model = model + '_kt' + args.keypoint_threshold.replace('.', '')  # 添加关键点阈值标记  
  
    # ==================== 路径配置 ====================  
    # COLMAP可执行文件路径  
    path_colmap = '/home/leon/repositories/colmap/build/src/exe/colmap'  
      
    # 根据数据集类型配置输入图像路径  
    if "Full_" in cluster:  
        # 完整视频测试数据  
        path_images = '/media/discoGordo/dataset_leon/UZ/test_frames' + '/' + video  
    elif "color_" in video:  
        # C3VD彩色数据集  
        path_images = '/media/discoGordo/C3VD/' + video + '/' + cluster  
    else:  
        # UZ基准测试数据  
        path_images = '/media/discoGordo/dataset_leon/UZ/colmap_benchmark_frames/' + video + '/' + cluster  
  
    # 配置输出结果路径，包含重投影误差参数  
    if args.reperror != '4':  
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '_err' + args.reperror + '/' + cluster  
    else:  
        path_results = '/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/' + model + '/' + video + '/' + cluster  
      
    # 数据库文件名（gm = geometric matching）  
    database_name = 'database_gm.db'  
      
    print("START")  
    # 创建输出目录，如果已存在则不报错  
    Path(path_results).mkdir(exist_ok=True, parents=True)  
      
    # ==================== 重建流程执行 ====================  
    # 开始计时  
    e0 = time.time()  
      
    # 步骤1: 特征提取和匹配（仅非SIFT方法）  
    if "sift_gm" not in model:  
        matches_extraction(path_images, path_results, args.featuretype, args.superpoint,   
                          args.matchtype, args.superglue, args.keypoint_threshold, args.overlap)  
        # 使用SuperGlue等深度学习方法生成匹配对  
      
    # 步骤2: 数据库初始化  
    initialize_database(database_name, path_images, path_results, args.matchtype, args.overlap)  
    # 创建COLMAP数据库，提取SIFT特征，初始化相机参数  
      
    # 打印数据库统计信息进行验证  
    db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)  
    print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),  
          len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))  
    db.close()  
  
    # 步骤3: 保存匹配结果到数据库（仅非SIFT方法）  
    if "sift_gm" not in model:  
        Save_database(database_name, path_images, path_results, args.reperror,   
                     args.matchtype, args.overlap)  
        # 将深度学习方法生成的匹配结果导入COLMAP数据库  
          
        # 再次打印统计信息验证数据导入  
        db = dataFunctions.COLMAPDatabase.connect(path_results + '/' + database_name)  
        print(len(db.get_all_cameras()), len(db.get_all_images()), len(db.get_all_keypoints()),  
              len(db.get_all_descriptors()), len(db.get_all_matches()), len(db.get_all_two_view_geometries()))  
        db.close()  
  
    # 步骤4: 三角化重建  
    triangulate(database_name, path_images, path_results, args.reperror, args.matchtype,   
               args.maxdistance, args.overlap, args.minsize)  
    # 执行COLMAP增量式SfM重建，生成3D点云和相机位姿  
      
    # ==================== 结果处理和日志记录 ====================  
    # 输出总执行时间  
    print("Time COLMAP: " + str(time.time() - e0))  
      
    # 记录执行时间到日志文件  
    with open('/media/discoGordo/dataset_leon/reconstructions_MICCAI2023/times_log.txt', 'a+') as f:  
        f.write(model + " " + video + " " + cluster + " " + "Time COLMAP: " + str(time.time() - e0) + '\n')  
      
    # 为每个重建的模型提取相机位姿  
    for res in os.listdir(path_results):  
        if res.isdigit():  
            T = cm.print_camera_positions(path_results + '/' + res)  # 输出相机位置信息
```

<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">COLMAP通过以下机制筛选特征：</font>

+ **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">三角化角度约束</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">：最小8度角确保几何稳定性</font>
+ **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">重投影误差过滤</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">：剔除误差过大的点</font>
+ **<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">多视图一致性</font>**<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">：只有在多个视图中一致观测到的点才保留</font>

```python
# 调用COLMAP mapper命令并传递关键参数实现这些筛选机制
subprocess.run([path_colmap, 'mapper',
                '--database_path', path_results + '/' + database_name,
                '--image_path', path_images,
                '--output_path', path_results,
                '--Mapper.init_min_tri_angle', '8',
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_extra_params', '0',
                '--Mapper.min_model_size', min_size,         # 三角化角度约束  
                '--Mapper.filter_max_reproj_error', error])  # 重投影误差过滤
```

| SfM 输出 | 后续作用 |
| --- | --- |
| 相机位姿（ 每一帧一个  ） | 3D→2D 投影 |
| 3D 点云 | 监督源 |
| Tracks 每一个 3D 点，在哪些图像帧中被观测到   | 构造Tracking loss 的 T |


 在 COLMAP 中，一个 track 表示同一个三维点在多个图像帧中的二维观测集合。作者利用 SfM 的几何一致性，只保留能够被多帧稳定观测并成功三角化的点作为可靠轨迹，这些轨迹随后被用作 Tracking Adaptation 的监督信号。  

### 3.3 从三维点到二维监督的映射过程
在获得三维点云及相机位姿之后，作者将每一个三维点重新投影到对应视频的各个图像帧中。这样，对于同一个三维点，可以在时间序列中得到一条由多个二维投影位置组成的轨迹。

在具体实现中，如果某一帧中该三维点对应的位置被特征检测器真实检测到，则该点被视为“观测到的点”；如果未被检测到，但其投影位置仍然落在图像范围内，则该位置仍然被保留为潜在的监督信息。通过这种方式，作者将稀疏的三维重建结果转化为可用于训练的二维监督信号。即**检测监督 Y**： 在某一帧 Ia 中： 所有能投影到 Ia 的 3D 点 构成一个 二维关键点集合。

<!-- 这是一张图片，ocr 内容为：从 3D 点到 2D 监督信号的构建流程(TRACKING ADAPTATION) 输入: -3D点云POINTS3D COLMAP 三维重建 相机位姿(RT.T.T) 相机内参 该点已被COLMAP 选取一个3D点XJ 成功三角化, 说明是真实可靠的物理点 逼历视频中的每一帧T 读取第T 帧相机位姿(R.T.T.T) 世界坐标>相机坐标 XCRT*XI+TT 鱼眼模型 应用相机投影模型 畸变校正 内参矩阵 理论上 该点在这一帧 得到2D投影位置(U_T,V.T) 应该出现的位置 COLMAP中 香 该帧是否 真实观测到? 弱监督: 强监督: VISIBLE[T]TRUE VISIBLE[T]FALSE 点存在但未被检测到 该点确实被检测到 还有下一帧? (POINTS 2D[T],VISIBLE[T]) 保存该3D点的完整时间轨迹 组成一条轨迹 作为 TRACKING ADAPTATION 的监督信号 -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766745802781-d2560aa7-773c-4d82-bc6c-d253d21173d5.png)

```python
# 获取3D点总数（跳过前3行头部信息）  
tope = len(pointsl)  
# 初始化投影数组：[图像数量, 3D点数量, 3] -> (x, y, z_camera坐标)  
points2d = np.zeros((len(images_ids), tope-3, 3), dtype=float)  
# 初始化可见性数组：标记每个3D点在每帧中是否被真实观测到  
visible = np.full((len(images_ids), tope-3), False)  
  
# 遍历所有3D点（从第4行开始，跳过文件头部）  
for i in range(3, len(pointsl)):  
    if i == tope:  
        break  
    line = pointsl[i].split()  
    # 提取3D点世界坐标 [X, Y, Z]  
    X = [float(x) for x in line[1:4]]  
      
    # 对每个图像帧进行投影  
    for ind in range(len(images_ids)):  
        im = images_ids[ind]  
        # 提取相机姿态的四元数表示  
        Q = images[im][0:4]  
        # 转换为旋转矩阵  
        R = qvec2rotmat(Q)  
        # 提取平移向量  
        T = images[im][4:7]  
        # 世界坐标 -> 相机坐标变换  
        Xc = R @ X + T  
        # 保存相机坐标系下的Z值（深度）  
        points2d[ind, i-3, 2] = Xc[2]  
          
        # 鱼眼相机模型投影  
        # 计算径向距离  
        r = np.sqrt(Xc[0] ** 2 + Xc[1] ** 2)  
        # 计算方位角  
        phi = np.arctan2(Xc[1], Xc[0])  
        # 计算极角  
        theta = np.arctan2(r, Xc[2])  
        # 应用畸变校正（多项式模型）  
        d = theta + k1 * theta ** 3 + k2 * theta ** 5 + k3 * theta ** 7 + k4 * theta ** 9  
        # 相机内参矩阵  
        K_c = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])  
        # 构建归一化坐标  
        x_c = np.array([[d * np.cos(phi), d * np.sin(phi), 1]]).T  
        # 投影到像素坐标  
        u = K_c @ x_c  
        # 保存2D投影坐标  
        points2d[ind, i-3, :2] = u[:2, 0]  
  
    # 处理COLMAP的观测信息，标记真实观测到的点  
    img_index = -1  
    for ind in range(len(line[8:])):  
        if ind % 2 == 0:  
            # 图像ID  
            img_id = int(line[8+ind])  
            img_index = images_ids.index(img_id)  
        else:  
            # 该图像中确实观测到了这个3D点  
            visible[img_index, i-3] = True  
  
# 保存投影结果和可见性信息  
np.savez_compressed(dst+"/points_projected",  
                    points2d=points2d,visible=visible,names=names)
```

**可靠轨迹（Reliable Track）的定义与作用**  
作者指出，并非所有由重建得到的轨迹都同样可靠。为此，本文提出仅使用“可靠轨迹”作为监督。**所谓可靠轨迹，是指同一个三维点在时间序列中形成的一条轨迹，其起始帧和终止帧均能够被特征检测器真实检测到。**

这一约束的直观含义是：即使特征点在中间若干帧中由于遮挡、反光或形变而暂时消失，只要它能够在时间前后再次稳定出现，该轨迹仍然被认为是可信的。通过这一筛选策略，作者有效降低了误匹配和噪声对训练过程的干扰。

<!-- 这是一张图片，ocr 内容为：(C) A FIG. 1. SUPERVISION POINTS OBTAINED FROM A COLMAP RECONSTRUCTION. ( ION. (A) ALL 3D POINTS 2 REPROJECTED INTO EACH VIDEO FRAME. WE DISTINGUISH POINTS THAT WERE ORIGINALLY CE DE- ARE I TECTED IN THIS FRAME (GREEN) AND POINTS THAT WERE NOT (BLUE), (B-D) ANALYZE A COMPLETE POINT TRACK, I.E., ALL THE POSITIONS OF THE SAME 3D POINT ALONG THE SEQUENCE. THE R E RE- T.(B) THE TRACK STARTS WHEN A POINT IS LIABLE TRACK FOR THIS POINT IS THE S THE GREEN SEGMENT. : (C) MOVEMENT OF THE POINT ALONG THE VIDEO. (D) WHEN THE FEATURE FIRST DETECTED. IS NOT DETECTED ANYMORE (E.G., BECAUSE , BECAUSE OF OCCLUSION), IT IS DEPICTED IN BLUE FROM THEN ON. -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766727693550-76f4a78b-f294-4e97-96cf-2a3824f578df.png)

 图 1 展示了本文如何从 COLMAP 的三维重建结果中构造特征点学习的监督信号。(a) 所有三维点被重新投影到每一帧图像中，其中绿色点表示该点在该帧中被真实检测到，蓝色点表示该点在该帧中未被检测到但通过三维投影得到。(b)-(d) 展示了同一个三维点在视频序列中的完整轨迹。当该点首次被检测到时，轨迹开始；随后随着视频推进，该点在图像中的位置发生变化；当该点由于遮挡等原因无法被检测到时，其轨迹以蓝色表示。本文仅使用起点和终点均为绿色的轨迹作为可靠轨迹，用于训练特征点网络。  

### 3.4 Tracking Adaptation 的训练方式
在训练阶段，作者不再使用“同一张图像 + 人工几何变换”的方式生成训练样本，而是直接从同一段内窥镜视频中采样不同时刻的两帧图像作为输入。网络需要在真实的时间变化和非刚性形变条件下，学习在不同帧之间保持特征点的一致性。

监督关系由可靠轨迹提供：属于同一轨迹的特征点对被视为正样本，而来自不同轨迹的特征点对则被视为负样本。**这种训练方式直接将“是否能够被稳定跟踪”作为学习目标，使网络更贴合 SfM 的实际需求。**

**损失函数设计与优化目标**  
整体损失函数由两部分组成。第一部分是检测损失，其形式与原始 SuperPoint 相同，但监督信号来源于三维重建后投影得到的特征点位置，而非合成的单应变换结果。第二部分是跟踪损失，作者采用基于 Triplet Loss 的形式，对特征描述子进行约束，使得同一可靠轨迹中的描述子在特征空间中彼此接近，而不同三维点对应的描述子保持足够区分。

superpoint原来的损失：该论文将Ld换成了Lt

<!-- 这是一张图片，ocr 内容为：(X,X,D,D,D,D;Y,Y',S)-LP(X,Y)+ (X',Y')+XLA(D,D',S ) ,S ) , (1) LSP( -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766746770862-f9be16fc-a1ba-44f8-b017-469b32cb88ea.png)

通过联合优化这两部分损失，网络被同时约束在两个层面上：一方面能够在单帧中检测到稳定的特征点，另一方面能够在时间维度上保持特征描述的一致性。最终，作者通过 Tracking Adaptation 训练得到的 SuperPoint-E，在内窥镜三维重建任务中表现出更高的稳定性和重建质量。

 在 Tracking Adaptation 中，由三维点重投影得到的二维关键点集合被进一步转换为 cell 级 65 维 one-hot 伪标签，以适配 SuperPoint 的检测头输出结构。检测损失 Lp 的形式与原始 SuperPoint 完全一致，仍采用基于 softmax 的交叉熵损失，不同之处仅在于**监督信号的来源。**  

<!-- 这是一张图片，ocr 内容为：HE,WC 了 UP(XHW;YHW), LP(X,Y) H W. WHERE EXP(XHWY) LP(XHW)LOG 65 HTTP://XHWK) -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766934795549-9899c448-b448-4b77-8d46-10900875f195.png)

（2）X为网络预测，Y为监督信号，Hc=H/8：cell 的行数。Wc=W/8：cell 的列数，lp为 第 (h,w) 个 cell 的分类损失  

（3）内层为 softmax 概率  ，外层为交叉熵损。如果预测概率高 → loss 小，如果预测概率低 → loss 大。

```python
def detector_loss(self, input, target, mask=None, loss_type="softmax"):  
    """  
    对检测器应用损失函数，默认使用softmax损失  
    :param input: 网络预测输出  
        tensor [batch_size, 65, Hc, Wc] - 65通道热力图预测  
    :param target: 从标签构建的目标  
        tensor [batch_size, 65, Hc, Wc] - 65通道one-hot标签  
    :param mask: 图像中的有效区域  
        tensor [batch_size, 1, Hc, Wc] - 掩码标记有效像素  
    :param loss_type: 损失类型  
        str (l2 or softmax) - softmax是原始论文使用的默认类型  
    :return: 归一化的损失值  
        tensor - 标量损失张量  
    """  
    if loss_type == "l2":  
        # L2损失：直接计算预测和目标的均方误差  
        loss_func = nn.MSELoss(reduction="mean")  
        loss = loss_func(input, target)  
    elif loss_type == "softmax":  
        # Softmax损失：使用BCE损失配合softmax激活  
        loss_func_BCE = nn.BCELoss(reduction='none').cuda()  
        # 对65通道预测应用softmax，转换为概率分布  
        loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)  
        # 应用掩码：只计算有效区域的损失  
        loss = (loss.sum(dim=1) * mask).sum()  
        # 归一化：按有效像素数量平均损失  
        loss = loss / (mask.sum() + 1e-10)  
    return loss
```

```python
# 创建交叉熵损失函数，设置reduce=False以保持每个像素的损失值  
loss_func = nn.CrossEntropyLoss(reduce=False).to(device)  
  
# 注释掉的代码：使用高斯标签时的BCE损失实现  
# if self.config['data']['gaussian_label']['enable']:  
#     loss = loss_func_BCE(nn.functional.softmax(semi, dim=1), labels3D_in_loss)  
#     loss = (loss.sum(dim=1) * mask_3D_flattened).sum()  
# else:  
# 计算交叉熵损失：semi是网络输出的65通道热力图，labels3D_in_loss是目标标签  
loss = loss_func(semi, labels3D_in_loss)  
  
# 应用掩码：只计算有效区域的损失，忽略边框等无效区域  
loss = (loss * mask_3D_flattened).sum()  
  
# 归一化：按有效像素数量平均损失，确保损失值不受有效像素数量变化影响  
loss = loss / (mask_3D_flattened.sum() + 1e-10)
```



<!-- 这是一张图片，ocr 内容为：LT(DA,DB,T) 14 (DAZ, DOJ, J), J), TI 1 J 1 F ATMAX(0 MP - DE, DOY) IF I, J, LT (DAZ, DOJ, J) (3) WITH MAX(0,,D DO,-MN)IFISJ -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766746783428-a9a3b296-8aee-4a25-8c46-602879cda666.png)

Da,Db： 在两帧图像 Ia 和 Ib 中，**与 tracks 对应的关键点的描述子集合** 

T：tracks 集合  Ll是一个**距离型损失, 描述子越像，loss 越小        ∣**T**∣**：两帧中共同可见的track数量

dai：第 i 条 track 在 **图像 a** 中的描述子，dbj：第 j 条 track 在 **图像 b** 中的描述子

 <!-- 这是一张图片，ocr 内容为：DU; -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766982877064-bf13cec6-397c-492f-9473-d659a69db562.png) 两个描述子的 **内积相似度， 因为描述子是 L2-normalized：  **<!-- 这是一张图片，ocr 内容为：DU;COS(0)E [-11,1] AI -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766982908014-294d6fa7-5af0-4e53-921f-7f0846df703c.png)

i=j（正样本）  （同一条 track，同一个 3D 点，不同帧的投影 ）mp：positive margin 一个超参数，表示：正样本至少应该有多相似

i≠j  (负样本）  （不同 track，不同 3D 点，不应该匹配 ）mn：negative margin负样本的“容忍上限”，希望不同点的相似度不超过 mn

```python
def descriptor_loss_tracking(descriptors1, descriptors2, correspondences, mask_valid=None,  
                             cell_size=8, device='cpu', descriptor_dist=4, lamda_d=250,  
                             num_matching_attempts=1000, num_masked_non_matches_per_match=10,  
                             dist='cos', method='1d', **config):  
    """  
    考虑批处理的描述子损失函数，专门用于tracking训练  
    :param descriptors1:  
        来自描述子头的输出，第一帧图像的描述子  
        tensor [descriptors, Hc, Wc]  
    :param descriptors2:  
        来自描述子头的输出，第二帧图像的描述子    
        tensor [descriptors, Hc, Wc]  
    """  
  
    def uv_to_tuple(uv):  
        """将(u,v)坐标张量转换为元组格式，便于索引操作"""  
        return (uv[:, 0], uv[:, 1])  
  
    def tuple_to_uv(uv_tuple):  
        """将元组格式的坐标转换回张量格式"""  
        return torch.stack([uv_tuple[0], uv_tuple[1]])  
  
    def tuple_to_1d(uv_tuple, W, uv=True):  
        """  
        将2D坐标元组转换为1D索引  
        :param uv: True表示使用(u,v)格式，False表示使用(v,u)格式  
        """  
        if uv:  
            return uv_tuple[0] + uv_tuple[1]*W  # u + v*W (列优先)  
        else:  
            return uv_tuple[0]*W + uv_tuple[1]  # v*W + u (行优先)  
  
    def uv_to_1d(points, W, uv=True):  
        """  
        将点坐标数组转换为1D索引  
        支持批量处理多个点  
        """  
        if uv:  
            return points[..., 0] + points[..., 1]*W  # u + v*W  
        else:  
            return points[..., 0]*W + points[..., 1]  # v*W + u
```

```python
def get_triplet_loss(image_a_pred, image_b_pred, matches_a, matches_b, non_matches_a, non_matches_b, alpha):  
    """  
    计算 Triplet Loss 损失函数  
      
    损失公式：\sum_{triplets} ||D(I_a, u_a, I_b, u_{b,match})||_2^2 - ||D(I_a, u_a, I_b, u_{b,non-match)||_2^2 + alpha   
      
    其中：  
    - D(I_a, u_a, I_b, u_{b,match}) 是匹配点对的描述子距离  
    - D(I_a, u_a, I_b, u_{b,non-match}) 是非匹配点对的描述子距离  
    - alpha 是间隔参数，确保正负样本对之间有足够间隔  
    """  
      
    # 获取匹配和非匹配点对的数量  
    num_matches = matches_a.size()[0]  
    num_non_matches = non_matches_a.size()[0]  
      
    # 计算扩展倍数，使匹配点对数量与非匹配点对数量一致  
    multiplier = num_non_matches / num_matches  
  
    ## 数据准备说明：  
    ## non_matches_a 已经被扩展到正确大小  
    ## non_matches_b 也已经被扩展  
    ## matches_a 是 non_matches_a 的较小版本  
    ## 只有 matches_b 需要被扩展到正确大小  
  
    # 将 matches_b 扩展到与非匹配点对相同的数量  
    # 先重复 multiplier 次，然后转置并展平为一维  
    matches_b_long = torch.t(matches_b.repeat(multiplier, 1)).contiguous().view(-1)  
                           
    # 从图像A的预测中提取匹配点a的描述子（使用扩展后的索引）  
    matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)  
      
    # 从图像B的预测中提取匹配点b的描述子（使用扩展后的索引）  
    matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b_long)  
      
    # 从图像B的预测中提取非匹配点b的描述子  
    non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)  
  
    # 计算 Triplet Loss：  
    # 正样本对距离 - 负样本对距离 + alpha  
    # 目标是让正样本对距离尽可能小，负样本对距离尽可能大  
    triplet_losses = (matches_a_descriptors - matches_b_descriptors).pow(2) - \  
                     (matches_a_descriptors - non_matches_b_descriptors).pow(2) + alpha  
      
    # 应用 hinge loss：只保留大于0的损失值，然后求平均  
    triplet_loss = 1.0 / num_non_matches * torch.clamp(triplet_losses, min=0).sum()  
  
    return triplet_loss
```

## 4 实验
本节实验的核心目的只有一个： **验证 Tracking Adaptation 训练出的 SuperPoint-E 是否真的更适合内窥镜三维重建（SfM）任务，而不仅仅是“检测到更多点”。**

### 4.1数据集与实验设置
作者使用的是 **EndoMapper 数据集**，该数据集来自真实的临床结肠镜检查，而非合成数据。

这一点非常重要，因为作者的方法正是为了解决**真实内窥镜视频中的困难场景**。

+ 数据来源：真实结肠镜视频
+ 场景特点：
    - 表面纹理极少
    - 强烈高光反射
    - 非刚性形变频繁

数据划分如下：

+ **训练集**：
    - 11260 帧图像
    - 65 个可成功重建的视频片段
+ **测试集**：
    - 838 帧图像
    - 7 个独立的视频片段

作者强调：

训练与测试视频在时间和患者层面上是相互独立的，以避免数据泄露。

### 4.2 对比方法与实验变量
为了公平评估 SuperPoint-E 的效果，作者设置了多组对比方法。

**4.2.1 传统与学习型基线方法**

包括：

+ **SIFT + COLMAP**
    - 作为传统 SfM 系统的代表
    - 使用手工特征点
+ **SuperPoint + SuperGlue + COLMAP**
    - 作为学习型特征点的直接迁移方案

这些方法用于回答一个基本问题：

直接使用已有特征点方法，在内窥镜场景下能做到什么程度？

**4.2.2 SuperPoint-E 的不同训练版本（消融实验）**

作者还设计了多种 SuperPoint-E 变体，用于分析 Tracking Adaptation 中各个设计的作用，包括：

+ 是否使用 Tracking Loss
+ 使用不同轨迹长度（Tr-N）进行训练
+ 不同监督配置下的性能变化

这些实验用于回答：

SuperPoint-E 的性能提升，是否真的来自 Tracking Adaptation？

### 4.3 评估指标
作者使用的评估指标**全部围绕 SfM 重建质量**，而非单纯的检测精度。

主要指标包括：

1. **∥3DIm∥**  
成功加入三维重建的图像比例。
    - 反映特征点是否足够稳定，能支撑相机位姿估计。
2. **∥3DPts∥**  
最终重建得到的三维点数量。
    - 反映 SfM 的几何密度。
3. **Err**  
所有三维点的平均重投影误差。
4. **Err-10K**  
重投影误差最小的 10000 个三维点的平均误差。
    - 用于衡量**高质量三维点**的精度。
5. **len(Tr)**  
特征点轨迹的平均长度。
    - 直接反映特征点在时间上的可跟踪性。

作者刻意强调：

这些指标共同衡量的是“是否有利于三维重建”，而不是“检测了多少点”。

### 4.4 消融实验结果（Table 1）
Table 1 展示了不同 SuperPoint-E 训练配置的对比结果。

<!-- 这是一张图片，ocr 内容为：SUPERVISION&TRAIN CONFIG. RECONSTRUCTION TEST RESULTS LEN(TR) 113DIMLL13DPTS ERR ERR-10K LOSS MATCH POINT 93.9% SP SP-O SP 6421.31.47 1.47 与 6.86 (ORIGINAL) SP (ORIGINAL) SF* 97.3% 8.39 SP-E VO 12707.9 1.50 1.66 SP-E V1 TR 8.95 1.69 1.51 13255.1 98.6% SF TR-2 9.45 99.1% LSP* SP-E V1 SF* TR 1.13 1.74 28308.3 TR-2 *士SP* SP-E V2 SF* TR 9.53 99.1% 1.75 1.02 34838.0 TR-N SF*+SP*H+TR SP + TR-N 99.2% 1.74 SP-E V3 30777.6 1.09 9.65 (BASE POINT DETECTOR):SP-O:ORIGINAL SUPERPOINT DETECTOR; SF*/SP*: SIFT/SP POINTS THAT POINT CHE COLMAP OPTIMIZATION, REPROJECTED IN EN D IN EACH VIDEO FRAME. WERE SUCCESSFULLY RECONSTRUCTED AFTER THE CO MATCH (MATCHES SUPERVISION):H:H:HON FROGRAPHY BASED, I.C, HOMOGRAPHIC ADAPTATION FROM ORIGINAL SUPERPOINT WORK; TR: THE PROPOSED TRACKING ADAPTATION. LOSS (LOSS USED FOR TRAINING): SP: ORIGINAL SUPERPOINT TRAININING LOSS; TI-2 OR TRACK-BASED LOSS TI-2 MEANS THAT THE LOSS IS COMPUTED FOR EVERY PAIR OFIMAGES IN THE TRACK. TI-N MEANS WE OPTIMIZE SIMULTANEOUSLY N VIEWS OF THE TRACK (N K(N IN OUR EXPERIMENTS). TABLE 1. ABLATION STUDY. CONFIGURATION OF THE TRAINING (L (LEFT),AND AVERAGE RECONSTRUC (RIGHT). BEST RESULTS HIGHLIGHTED IN BOLD. TION RESULTS, I.E.,QUALITY METRICS -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766739064099-aace2f6f-2dd0-49e6-8746-251c77e2bc56.png)

核心结论包括：

+ **引入 Tracking Adaptation 后**：
    - 重建的三维点数量显著增加
    - 特征点轨迹长度明显变长
+ **多帧联合监督（Tr-N）**：
    - 相比只使用两帧监督
    - 能进一步提升重建稳定性

消融实验对比了基于轨迹的两种训练方式：Tr-2 仅在成对图像之间施加描述子一致性约束，而 Tr-N 同时对同一轨迹中的多视图进行联合优化。实验表明，多视一致性约束能够显著提升特征在时间维度上的稳定性，更符合 SfM 对特征点的实际需求。  

作者指出：

使用 Tracking Adaptation 的模型，在所有 SfM 相关指标上均优于不使用该策略的版本。

其中表现最好的模型为 **SP-E v2**：

+ ∥3DPts∥ 最大
+ Err-10K 最小

### 4.5 与基线方法的整体对比（Table 2）
<!-- 这是一张图片，ocr 内容为：SUBSEQUENCE 001_1002_1 016_1017_1095_1 (STD) 014_1 095_2 AVG RECONSTRUCTEDIMAGES(|3DIMLL) 155 107 TOTAL 109 125118 105 119.7 (15.9) 119 716%100%520%975%9900% 98.1% 91.6% SIFT 87.1% (17.0) 100% 100%93.6% 100% 89.6% 100% 97.5% SP (3.9) 93.6% 100% 100% 100% 100% 100% 100% SP-E (OURS) (2.3) 99.1% RECONSTRUCTED POINTS 3DPTS SIFT (7237.6) 10253.3 2505 13470 7666 5700 9608 26225 6599 4133.1) 10211.0 SP 4093 12941 17451 6489 891112535 9057 34851 45471 42727 33277 36403 19286 31851 34838.0 SP-E(OURS) (7846.5) (ERR) MEAN REPROJECTION N ERROR SIFT 1.38 1.40 1.45 1.31 (0.15) 0.95 1.34 1.34 1.30 SP 1.52 1.58 1.50 1.38 1.49 1.51 (0.06) 1.48 1.51 SP-E(OURS) 1.69 1.71 (0.07) 1.90 1.75 1.81 1.68 1.75 1.73 ERR-10K) MEAN REPROJECTION ERROR OF THE BEST 10K POINTS SIFT 1.45 1.20 (0.32) 1.08 1.30 1.34 1.40 1.38 0.46 SP 1.48 1.58 (0.19) 1.38 1.00 1.30 1.51 1.30 1.49 SP-E(OURS) 0.92 1.02 1.06 1.41 1.30 0.73 0.91 (0.23) 0.84 MEAN TRACK LENGTH (LEN(TR)) SIFT 2.70) 9.12 7.74 7.56 6.57 10.88 12.48 12.88 5.73 SP 5.38 5.16 (1.59) 8.20 6.49 4.52 5.54 7.86 8.73 (2.55) 8.78 SP-E(OURS)7.05 6.78 9.6 11.29 9.53 9.63 14.73 8.42 TOTAL NUMBER OF IMAGES IN THE SUBSEQUENCE. * IF 10K POINTS ARE NOT AVAILABLE, AVERAGE IS COMPUTED OVER A ALL AVAILABLE RECONSTRUCTED POINTS. E COMPARISON TO THE BASELINES. TABLE 2. RECONSTRUCTION QUALITY METRICS FOR THE COT -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766739099493-041b7325-8030-4128-9063-c80fd07c99ba.png)

Table 2 对比了：

+ SIFT
+ 原始 SuperPoint
+ SuperPoint-E

在相同 COLMAP 重建流程下的表现。

主要观察结果为：

+ **SuperPoint-E 重建的三维点数量是基线方法的 3 倍以上**
+ 几乎所有测试图像都能被成功加入重建
+ 在高质量点（Err-10K）指标上误差最低

这说明：

SuperPoint-E 不仅检测到更多点，而且这些点确实被 SfM 系统有效利用。

### 4.6 特征点空间分布分析（Table 3）
<!-- 这是一张图片，ocr 内容为：SPREAD OF FEATURES 个 OF FEATURES ON SPECULARITIES SIFT 43.9% 28.6% 19.6% SP 56.9% SP-E (OURS) 67.5% 9.9% TABLE 3. ANALYSIS OF THE FEATURE LOCATIONS FOR EACH ME CH METHOD. -->
![](https://cdn.nlark.com/yuque/0/2025/png/60748161/1766739128115-e0ec4871-509e-44cc-b355-b4f03c7f1a61.png)

作者进一步分析了不同方法检测到的特征点在图像中的空间分布。

结果表明：

+ 原始 SuperPoint 和 SIFT：
    - 特征点大量集中在反光区域
    - 这些点往往不稳定
+ **SuperPoint-E**：
    - 特征点分布更加均匀
    - 显著减少了落在强反光区域的比例

作者据此指出：

Tracking Adaptation 使网络学会回避内窥镜中不可靠的区域，从而提升整体重建质量。

**实验小结**

通过系统性的实验，作者证明：

**使用真实 SfM 成功轨迹作为监督信号，可以显著提升特征点在内窥镜三维重建任务中的有效性。**

这为本文提出的 Tracking Adaptation 提供了直接且有力的实验证据。





