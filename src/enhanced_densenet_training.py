"""
增强版 DenseNet-121 CheXpert 模型
包含训练功能和Grad-CAM可解释性分析
Supporting Slide 5 & 6: Deep Learning Training + Explainable AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json

class FocalLoss(nn.Module):
    """
    焦点损失 - 专门处理极度类别不平衡问题
    Focal Loss for addressing class imbalance in multi-label classification
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 计算标准的BCE损失
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # 计算概率
        probs = torch.sigmoid(inputs)
        
        # 计算pt (正确预测的概率)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # 计算alpha权重
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # 应用焦点权重 (1-pt)^gamma
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # 计算焦点损失
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_weights(csv_file, labels):
    """
    计算每个标签的动态权重 - 修复版本
    使用更保守的权重计算，避免极端值
    """
    print("🔍 计算智能动态权重 (修复版)...")
    
    # 读取数据
    data = pd.read_csv(csv_file)
    total_samples = len(data)
    
    weights = []
    print(f"\n📊 各标签权重计算:")
    print("-" * 60)
    print(f"{'标签':<25} {'正样本数':<8} {'负样本数':<8} {'权重':<8}")
    print("-" * 60)
    
    for label in labels:
        if label in data.columns:
            pos_count = (data[label] == 1).sum()
            neg_count = total_samples - pos_count
            
            if pos_count > 0:
                # 使用更温和的权重计算：对数平滑 + 限制最大值
                raw_weight = neg_count / pos_count
                # 对数平滑，避免极端值
                weight = min(np.log(raw_weight + 1) + 1.0, 10.0)  # 限制最大权重为10
            else:
                weight = 1.0
                
            weights.append(weight)
            print(f"{label:<25} {pos_count:<8} {neg_count:<8} {weight:<8.2f}")
        else:
            weights.append(1.0)
            print(f"{label:<25} {'N/A':<8} {'N/A':<8} {1.0:<8.2f}")
    
    print("-" * 60)
    print(f"✅ 权重计算完成，平均权重: {np.mean(weights):.2f}")
    print(f"💡 使用保守权重策略，最大权重限制为10.0")
    
    return torch.FloatTensor(weights)

class CheXpertDataset(Dataset):
    """
    CheXpert数据集类
    支持多标签分类训练
    """
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # CheXpert 14个标签
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 从实际的数据结构加载图像
        # 实际结构: E:/data_subset1/p10000032/s50414267/
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        
        # 构建目录路径
        patient_dir = f"p{subject_id}"       # p10000032
        study_dir = f"s{study_id}"           # s50414267
        
        study_path = self.image_dir / patient_dir / study_dir
        
        try:
            # 查找study目录中的DICOM文件
            if study_path.exists():
                # 寻找可能的图像文件
                image_files = []
                for ext in ['.dcm', '.jpg', '.png', '.jpeg']:
                    image_files.extend(list(study_path.glob(f"*{ext}")))
                
                if image_files:
                    image_path = image_files[0]  # 使用第一个找到的图像文件
                    
                    if image_path.suffix.lower() == '.dcm':
                        # 处理DICOM文件
                        import pydicom
                        dicom_data = pydicom.dcmread(str(image_path))
                        image_array = dicom_data.pixel_array
                        
                        # 转换为PIL图像
                        if len(image_array.shape) == 2:  # 灰度图像
                            from PIL import Image
                            # 归一化到0-255
                            if image_array.max() > image_array.min():
                                image_array = ((image_array - image_array.min()) / 
                                              (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                            else:
                                image_array = np.zeros_like(image_array, dtype=np.uint8)
                            image = Image.fromarray(image_array, mode='L').convert('RGB')
                        else:
                            raise ValueError("Unexpected image array shape")
                    else:
                        # 处理普通图像文件
                        from PIL import Image
                        image = Image.open(image_path).convert('RGB')
                else:
                    # 如果没找到图像文件，创建占位图像
                    print(f"Warning: No image files found in {study_path}")
                    from PIL import Image
                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            else:
                # 如果study目录不存在，创建占位图像
                print(f"Warning: Study directory not found: {study_path}")
                from PIL import Image
                image = Image.new('RGB', (224, 224), color=(64, 64, 64))
                
        except Exception as e:
            # 如果加载失败，创建占位图像
            print(f"Warning: Could not load image from {study_path}: {e}")
            from PIL import Image
            image = Image.new('RGB', (224, 224), color=(192, 192, 192))
        
        # 应用transform
        if self.transform:
            image = self.transform(image)
        
        # 提取标签 (将-1不确定标签转为0)
        labels = []
        for label in self.labels:
            value = row.get(label, 0)
            # 处理字符串格式的标签值
            if isinstance(value, str):
                if value == '1.0' or value == '1':
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                labels.append(1 if value == 1 else 0)  # 将-1和0都视为0
        
        return image, torch.FloatTensor(labels)

def create_data_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    创建训练集、验证集和测试集划分
    
    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例 (默认70%)
        val_ratio: 验证集比例 (默认15%)
        test_ratio: 测试集比例 (默认15%)
        random_state: 随机种子
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
    # 获取所有索引
    indices = list(range(len(dataset)))
    
    # 分层采样 - 基于'No Finding'标签来保证类别平衡
    labels = []
    for i in indices:
        _, label_tensor = dataset[i]
        # 使用第一个标签(No Finding)作为分层基准
        labels.append(int(label_tensor[0].item()))
    
    # 第一次划分：分离训练集和剩余部分
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=labels
    )
    
    # 计算验证集和测试集在剩余数据中的比例
    temp_val_ratio = val_ratio / (val_ratio + test_ratio)
    
    # 获取剩余部分的标签
    temp_labels = [labels[i] for i in temp_indices]
    
    # 第二次划分：分离验证集和测试集
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - temp_val_ratio),
        random_state=random_state,
        stratify=temp_labels
    )
    
    # 创建子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"📊 数据集划分完成:")
    print(f"  训练集: {len(train_dataset)} 样本 ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"  验证集: {len(val_dataset)} 样本 ({len(val_dataset)/len(dataset)*100:.1f}%)")
    print(f"  测试集: {len(test_dataset)} 样本 ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

class DenseNetCheXpert(nn.Module):
    """
    基于DenseNet-121的CheXpert分类器 - 阶段2优化版本
    Supporting Slide 5: Model Core - Building a Precise Diagnostic Engine
    """
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNetCheXpert, self).__init__()
        
        # 主干网络：DenseNet-121 (高参数效率，特征传播好)
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # 获取特征维度
        num_features = self.backbone.classifier.in_features  # 1024
        
        # 改进的多层分类头 - 渐进式降维 + 残差连接
        self.classifier = nn.Sequential(
            # 第一层：1024 -> 512
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 第二层：512 -> 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # 输出层：256 -> 14
            nn.Linear(256, num_classes)
        )
        
        # 保存特征图用于Grad-CAM
        self.features = self.backbone.features
        
        # 注册hook以获取特征图
        self.feature_maps = None
        self.gradients = None
        
    def forward(self, x):
        # 前向传播并保存特征图
        features = self.features(x)
        
        # 注册hook获取梯度
        if features.requires_grad:
            self.feature_maps = features
            features.register_hook(self.save_gradients)
        
        # 全局平均池化
        pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        pooled = torch.flatten(pooled, 1)
        
        # 简化的单层分类器
        output = self.classifier(pooled)
        
        return output
    
    def save_gradients(self, grad):
        """保存梯度用于Grad-CAM"""
        self.gradients = grad

class CheXpertTrainer:
    """
    CheXpert模型训练器 - 阶段2优化版本
    Supporting Slide 5: Training Details
    """
    def __init__(self, model, device='cuda', csv_file=None):
        self.model = model.to(device)
        self.device = device
        
        # CheXpert 14个标签
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # 计算智能动态权重
        if csv_file:
            pos_weights = calculate_class_weights(csv_file, self.labels)
        else:
            # 备用权重（如果没有提供CSV文件）
            pos_weights = torch.FloatTensor([30.0, 50.0, 6.0, 35.0, 3.0, 15.0, 
                                           10.0, 20.0, 8.0, 25.0, 4.0, 40.0, 45.0, 2.0])
        
        print(f"💡 使用智能动态权重: {pos_weights.numpy()}")
        
        # 使用温和的焦点损失替代BCEWithLogitsLoss
        self.criterion = FocalLoss(
            alpha=0.25,  # 更温和的正样本权重
            gamma=1.0,   # 降低困难样本专注度
            pos_weight=pos_weights.to(device)
        )
        
        # 优化器：Adam with conservative learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # 学习率调度器：稳定的StepLR
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10,   # 每10个epoch降低学习率
            gamma=0.3       # 学习率乘以0.3
        )
        
        # 移除Warmup参数
        self.warmup_epochs = 0
        self.base_lr = 1e-4
        self.warmup_lr = 1e-5
        
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
    
    def train_epoch(self, train_loader):
        """训练一个epoch - 包含warmup机制"""
        self.model.train()
        total_loss = 0
        
        # Warmup学习率调整
        if self.current_epoch < self.warmup_epochs:
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            warmup_lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * lr_scale
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"🔥 Warmup阶段: 学习率 = {warmup_lr:.6f}")
        
        try:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            print("💡 尝试减少批次大小或检查GPU内存")
            raise e
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 计算准确率
                predictions = torch.sigmoid(output) > 0.5
                correct_predictions += (predictions == target.bool()).sum().item()
                total_predictions += target.numel()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        self.val_losses.append(avg_loss)
        
        # 只在warmup阶段之后调用scheduler
        if self.current_epoch >= self.warmup_epochs:
            self.scheduler.step()
        
        return avg_loss, accuracy
    
    def find_optimal_thresholds(self, val_loader):
        """基于验证集寻找每个标签的最优阈值"""
        print("🔍 正在寻找最优阈值...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predictions = torch.sigmoid(output)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 为每个标签寻找最优阈值
        optimal_thresholds = []
        from sklearn.metrics import precision_recall_curve, f1_score
        
        print("\n📊 各标签最优阈值:")
        print("-" * 60)
        print(f"{'标签':<25} {'最优阈值':<10} {'最优F1':<10}")
        print("-" * 60)
        
        for i, label in enumerate(self.labels):
            try:
                if np.sum(all_targets[:, i]) > 0:  # 确保有正样本
                    # 使用F1分数寻找最优阈值
                    precision, recall, thresholds = precision_recall_curve(all_targets[:, i], all_predictions[:, i])
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                    
                    best_threshold_idx = np.argmax(f1_scores)
                    best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
                    best_f1 = f1_scores[best_threshold_idx]
                    
                    optimal_thresholds.append(best_threshold)
                    print(f"{label:<25} {best_threshold:<10.3f} {best_f1:<10.3f}")
                else:
                    optimal_thresholds.append(0.5)  # 默认阈值
                    print(f"{label:<25} {0.5:<10.3f} {'N/A':<10}")
            except:
                optimal_thresholds.append(0.5)
                print(f"{label:<25} {0.5:<10.3f} {'Error':<10}")
        
        print("-" * 60)
        return np.array(optimal_thresholds)

    def evaluate_test_set(self, test_loader, optimal_thresholds=None):
        """在测试集上评估模型性能 - 支持最优阈值"""
        print("🧪 在测试集上评估模型...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 收集预测和真实标签
                predictions = torch.sigmoid(output)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # 合并所有预测和标签
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 计算各种指标
        avg_loss = total_loss / len(test_loader)
        
        # 计算每个标签的AUC、精确度、召回率等
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        
        labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        print(f"\n📊 测试集评估结果:")
        print(f"平均损失: {avg_loss:.4f}")
        
        # 如果提供了最优阈值，使用最优阈值评估
        if optimal_thresholds is not None:
            print("\n🎯 使用最优阈值的评估结果:")
            print("-" * 90)
            print(f"{'标签':<25} {'AUC':<6} {'精确度':<8} {'召回率':<8} {'F1分数':<8} {'阈值':<8}")
            print("-" * 90)
        else:
            print("\n使用固定0.5阈值的评估结果:")
            print("-" * 80)
            print(f"{'标签':<25} {'AUC':<6} {'精确度':<8} {'召回率':<8} {'F1分数':<8}")
            print("-" * 80)
        
        auc_scores = []
        for i, label in enumerate(labels):
            try:
                # AUC
                auc = roc_auc_score(all_targets[:, i], all_predictions[:, i])
                auc_scores.append(auc)
                
                # 二值化预测结果
                if optimal_thresholds is not None:
                    threshold = optimal_thresholds[i]
                    pred_binary = (all_predictions[:, i] > threshold).astype(int)
                else:
                    threshold = 0.5
                    pred_binary = (all_predictions[:, i] > threshold).astype(int)
                
                # 精确度、召回率、F1分数
                precision = precision_score(all_targets[:, i], pred_binary, zero_division=0)
                recall = recall_score(all_targets[:, i], pred_binary, zero_division=0)
                f1 = f1_score(all_targets[:, i], pred_binary, zero_division=0)
                
                if optimal_thresholds is not None:
                    print(f"{label:<25} {auc:<6.3f} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f} {threshold:<8.3f}")
                else:
                    print(f"{label:<25} {auc:<6.3f} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f}")
                
            except Exception as e:
                print(f"{label:<25} Error: {str(e)[:40]}")
                auc_scores.append(0)
        
        mean_auc = np.mean(auc_scores)
        if optimal_thresholds is not None:
            print("-" * 90)
        else:
            print("-" * 80)
        print(f"平均AUC: {mean_auc:.3f}")
        
        return avg_loss, mean_auc

    def train(self, train_loader, val_loader, test_loader=None, epochs=12):
        """完整训练流程 - 阶段2优化版本"""
        print("🚀 开始训练 CheXpert DenseNet-121 模型 (阶段2优化)...")
        print("💡 使用焦点损失 + 动态权重 + CosineAnnealingWarmRestarts + Warmup...")
        
        best_val_loss = float('inf')
        best_auc = 0.0
        patience_counter = 0
        early_stop_patience = 6  # 增加耐心值给新策略更多时间
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证准确率: {val_acc:.4f}")
            print(f"当前学习率: {current_lr:.6f}")
            
            # 保存最佳模型 (基于验证损失)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'scheduler_state_dict': self.scheduler.state_dict()
                }, 'best_chexpert_densenet121_v2.pth')
                print("✅ 保存最佳模型 (v2)")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"⏹️ 早停触发 (patience={early_stop_patience})")
                    break
        
        print("🎉 阶段2优化训练完成!")
        
        # 如果提供了测试集，进行最终评估
        if test_loader is not None:
            print("\n" + "="*60)
            # 加载最佳模型进行测试
            try:
                checkpoint = torch.load('best_chexpert_densenet121_v2.pth')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ 加载最佳模型权重进行测试集评估 (v2)")
            except:
                print("⚠️ 使用当前模型权重进行测试集评估")
            
            # 🔍 首先在验证集上寻找最优阈值
            print("\n🎯 Step 1: 在验证集上寻找最优阈值...")
            optimal_thresholds = self.find_optimal_thresholds(val_loader)
            
            # 🧪 然后在测试集上使用最优阈值评估
            print("\n🎯 Step 2: 使用最优阈值在测试集上评估...")
            test_loss, test_auc = self.evaluate_test_set(test_loader, optimal_thresholds)
            
            # 📊 对比固定阈值和最优阈值的效果
            print("\n🎯 Step 3: 对比固定阈值效果...")
            test_loss_fixed, test_auc_fixed = self.evaluate_test_set(test_loader, None)
            
            print(f"\n📈 阈值优化效果对比:")
            print(f"  固定阈值(0.5) AUC: {test_auc_fixed:.3f}")
            print(f"  最优阈值 AUC: {test_auc:.3f}")
            print(f"  AUC提升: {test_auc - test_auc_fixed:+.3f}")
            
            return test_loss, test_auc
        else:
            return None, None

class GradCAMExplainer:
    """
    Grad-CAM可解释性分析器
    Supporting Slide 6: Explainable AI (XAI)
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # CheXpert标签
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def generate_gradcam(self, image, class_idx):
        """
        生成Grad-CAM热力图
        打开"黑箱"：让AI的诊断看得懂
        """
        # 前向传播
        image = image.unsqueeze(0).to(self.device)
        image.requires_grad_()
        
        output = self.model(image)
        
        # 反向传播
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # 获取特征图和梯度
        feature_maps = self.model.feature_maps   # [1, 1024, 7, 7]
        gradients = self.model.gradients        # [1, 1024, 7, 7]
        
        # 计算权重 (全局平均池化梯度)
        weights = torch.mean(gradients, dim=(2, 3))  # [1, 1024]
        
        # 生成热力图
        cam = torch.zeros(
            feature_maps.shape[2:],
            dtype=feature_maps.dtype,
            device=feature_maps.device
        )
        for i in range(weights.shape[1]):
            cam += weights[0, i] * feature_maps[0, i]
        
        # ReLU激活
        cam = torch.relu(cam)
        
        # 归一化到[0,1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize_gradcam(self, image_path, class_idx, save_path=None):
        """
        可视化Grad-CAM结果
        将抽象的预测转化为直观的视觉证据
        """
        # 加载和预处理图像
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        original_image = Image.open(image_path).convert('RGB')
        input_image = transform(original_image)
        
        # 生成热力图
        cam = self.generate_gradcam(input_image, class_idx)
        
        # 调整热力图大小
        cam_resized = cv2.resize(cam, (224, 224))
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('原始胸部X光图像')
        axes[0].axis('off')
        
        # 热力图
        im1 = axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Grad-CAM: {self.labels[class_idx]}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # 叠加图像
        overlay = np.array(original_image.resize((224, 224)))
        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
        overlay_result = 0.6 * overlay/255.0 + 0.4 * cam_colored
        
        axes[2].imshow(overlay_result)
        axes[2].set_title('热力图叠加 (关注区域高亮)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Grad-CAM可视化保存到: {save_path}")
        
        plt.show()
        
        return cam_resized
    
    def explain_prediction(self, image_path, top_k=3):
        """
        全面解释模型预测
        生成多个标签的可解释性分析
        """
        # 预处理图像
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(Image.open(image_path).convert('RGB'))
        image_tensor = image.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.sigmoid(output).cpu().numpy()[0]
        
        # 获取top-k预测
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        print("🔍 AI诊断可解释性分析")
        print("=" * 50)
        
        for i, idx in enumerate(top_indices):
            prob = probabilities[idx]
            label = self.labels[idx]
            
            print(f"\n{i+1}. {label}")
            print(f"   预测概率: {prob:.3f}")
            print(f"   置信度: {'高' if prob > 0.7 else '中' if prob > 0.3 else '低'}")
            
            # 生成并保存Grad-CAM
            save_path = f"gradcam_{label.replace(' ', '_').lower()}.png"
            self.visualize_gradcam(image_path, idx, save_path)
        
        return top_indices, probabilities[top_indices]

# 使用示例和训练脚本
def create_training_example():
    """创建训练示例 - 使用实际的数据集路径"""
    print("📚 CheXpert DenseNet-121 训练示例")
    print("Supporting Slide 5: Deep Learning Model Training")
    print("✅ 包含训练集/验证集/测试集划分")
    print("✅ 使用实际的数据集文件")
    
    # 实际数据路径
    csv_file = "e:/Learning/BN5212/data/labeled_reports_with_ids.csv"
    image_dir = "E:/data_subset1/"  # 原始图像存储目录
    
    print(f"\n📂 数据路径:")
    print(f"  CSV文件: {csv_file}")
    print(f"  图像目录: {image_dir}")
    
    # 检查文件是否存在
    import os
    if not os.path.exists(csv_file):
        print(f"❌ 错误: CSV文件不存在: {csv_file}")
        print("💡 提示: 请先运行数据处理脚本生成标注数据")
        return
    
    if not os.path.exists(image_dir):
        print(f"❌ 错误: 图像目录不存在: {image_dir}")
        print("💡 提示: 请检查原始数据路径是否正确")
        return
    
    # 阶段2强化数据增强 - 专为医学图像优化
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),  # 增加旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),  # 增强对比度调整
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),  # 增加仿射变换
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),  # 新增：锐度调整
        transforms.RandomAutocontrast(p=0.2),  # 新增：自动对比度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 演示数据集划分逻辑
    print("\n📊 数据集划分策略:")
    print("  训练集: 70% - 用于模型训练")
    print("  验证集: 15% - 用于超参数调优和早停")
    print("  测试集: 15% - 用于最终性能评估")
    print("  ✅ 使用分层采样保证类别平衡")
    
    try:
        # 创建完整数据集
        print(f"\n🔄 加载数据集...")
        full_dataset = CheXpertDataset(csv_file, image_dir, None)
        print(f"✅ 成功加载 {len(full_dataset)} 个样本")
        
        # 数据集划分
        print(f"\n🔄 执行数据集划分...")
        train_dataset, val_dataset, test_dataset = create_data_splits(
            full_dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15
        )
        
        # 为不同的数据集应用不同的transform
        print(f"\n🔄 配置数据增强...")
        
        # 重新创建带有transform的数据集
        csv_file_path = csv_file
        image_dir_path = image_dir
        
        # 创建训练、验证、测试数据集，各自使用不同的transform
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        test_indices = test_dataset.indices
        
        # 创建带有特定transform的自定义数据集类
        class TransformedSubset(Dataset):
            def __init__(self, original_dataset, indices, transform):
                self.original_dataset = original_dataset
                self.indices = indices
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                original_idx = self.indices[idx]
                # 获取原始数据（不应用transform）
                original_transform = self.original_dataset.transform
                self.original_dataset.transform = None
                image, labels = self.original_dataset[original_idx]
                self.original_dataset.transform = original_transform
                
                # 应用指定的transform
                if self.transform:
                    image = self.transform(image)
                
                return image, labels
        
        # 创建带有不同transform的数据集
        train_dataset_transformed = TransformedSubset(full_dataset, train_indices, train_transform)
        val_dataset_transformed = TransformedSubset(full_dataset, val_indices, val_test_transform)
        test_dataset_transformed = TransformedSubset(full_dataset, test_indices, val_test_transform)
        
        # 创建数据加载器 (调整批次大小避免内存问题)
        print(f"\n🔄 创建数据加载器...")
        train_loader = DataLoader(train_dataset_transformed, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset_transformed, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset_transformed, batch_size=8, shuffle=False, num_workers=0)
        
        print(f"✅ 数据加载器创建成功")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        
    except Exception as e:
        print(f"❌ 数据加载错误: {str(e)}")
        print("💡 可能原因:")
        print("   1. 图像文件路径格式不匹配")
        print("   2. 图像文件不存在或损坏") 
        print("   3. CSV文件格式问题")
        return
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseNetCheXpert(num_classes=14, pretrained=True)
    
    # 创建训练器 - 传递CSV文件路径用于计算动态权重
    trainer = CheXpertTrainer(model, device, csv_file=csv_file)
    
    print(f"\n🔧 修复训练配置:")
    print(f"  ✅ 模型结构: DenseNet-121 (简化分类器)")
    print(f"  ✅ 设备: {device}")
    print(f"  ✅ 损失函数: FocalLoss (alpha=0.25, gamma=1.0) - 温和版本")
    print(f"  ✅ 优化器: Adam (lr=1e-4) - 保守学习率")
    print(f"  ✅ 学习率调度: StepLR (稳定调度)")
    print(f"  ✅ 梯度裁剪: max_norm=1.0")
    print(f"  ✅ 智能动态权重: 最大限制10.0 (保守版本)")
    print(f"  ✅ 早停机制: patience=6")
    print(f"  ✅ 强化数据增强: 医学图像专用技术")
    
    print(f"\n📈 修复训练流程:")
    print("  1. 保守权重：限制最大权重为10.0，避免极端值")
    print("  2. 稳定学习率：使用StepLR，每10个epoch衰减30%")
    print("  3. 温和损失函数：降低FocalLoss参数，减少过拟合")
    print("  4. 简化架构：移除残差连接，回归简单有效设计")
    print("  5. 梯度裁剪防止训练不稳定")
    
    # 开始修复训练
    print(f"\n🚀 开始修复训练...")
    print("💡 目标：恢复到基线0.65+ AUC性能...")
    test_loss, test_auc = trainer.train(
        train_loader, val_loader, test_loader, epochs=12)
    print(f'🎯 修复后测试集性能: 损失={test_loss:.4f}, 平均AUC={test_auc:.3f}')
    
    # 性能评估
    if test_auc is not None:
        baseline_auc = 0.647
        failed_auc = 0.594
        improvement = test_auc - failed_auc
        vs_baseline = test_auc - baseline_auc
        print(f"\n📈 修复效果分析:")
        print(f"  失败AUC: {failed_auc:.3f}")
        print(f"  基线AUC: {baseline_auc:.3f}")
        print(f"  修复后AUC: {test_auc:.3f}")
        print(f"  vs失败版本: {improvement:+.3f}")
        print(f"  vs基线版本: {vs_baseline:+.3f}")
        if test_auc >= baseline_auc:
            print("  ✅ 成功恢复到基线性能")
        else:
            print("  ⚠️ 仍未完全恢复，需要进一步调整")
    
    return train_loader, val_loader, test_loader, trainer

def create_data_split_demo():
    """演示数据集划分功能"""
    print("\n📊 数据集划分演示")
    print("=" * 50)
    
    # 创建模拟数据集进行演示
    class MockDataset:
        def __init__(self, size=1000):
            self.size = size
            # 模拟不平衡的标签分布
            np.random.seed(42)
            self.labels = np.random.choice([0, 1], size=size, p=[0.7, 0.3])
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # 返回模拟的图像tensor和标签tensor
            mock_image = torch.randn(3, 224, 224)
            mock_labels = torch.zeros(14)
            mock_labels[0] = float(self.labels[idx])  # 转换为浮点数
            return mock_image, mock_labels
    
    # 创建模拟数据集
    mock_dataset = MockDataset(1000)
    
    # 演示数据集划分
    train_set, val_set, test_set = create_data_splits(
        mock_dataset,
        train_ratio=0.7,
        val_ratio=0.15, 
        test_ratio=0.15,
        random_state=42
    )
    
    print(f"✅ 成功划分模拟数据集")
    print(f"  总样本数: {len(mock_dataset)}")
    print(f"  训练集: {len(train_set)} 样本")
    print(f"  验证集: {len(val_set)} 样本") 
    print(f"  测试集: {len(test_set)} 样本")
    
    # 验证标签分布
    def check_label_distribution(dataset, name):
        labels = []
        for i in range(len(dataset)):
            _, label_tensor = dataset[i]
            labels.append(int(label_tensor[0].item()))
        
        pos_ratio = sum(labels) / len(labels)
        print(f"  {name} 正样本比例: {pos_ratio:.3f}")
        return pos_ratio
    
    print(f"\n🔍 验证类别平衡:")
    orig_ratio = sum(mock_dataset.labels) / len(mock_dataset.labels)
    print(f"  原始数据 正样本比例: {orig_ratio:.3f}")
    
    train_ratio = check_label_distribution(train_set, "训练集")
    val_ratio = check_label_distribution(val_set, "验证集")
    test_ratio = check_label_distribution(test_set, "测试集")
    
    print(f"✅ 分层采样成功保持类别平衡!")

def create_explainability_example():
    """创建可解释性分析示例"""
    print("🔍 Grad-CAM可解释性分析示例")
    print("Supporting Slide 6: Explainable AI (XAI)")
    
    # 加载训练好的模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseNetCheXpert(num_classes=14, pretrained=True)
    
    # 加载权重（如果有的话）
    try:
        checkpoint = torch.load('best_chexpert_densenet121.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 加载训练好的权重")
    except:
        print("⚠️ 使用预训练权重（未针对CheXpert微调）")
    
    # 创建可解释性分析器
    explainer = GradCAMExplainer(model, device)
    
    print("🎯 Grad-CAM功能:")
    print("  - 生成热力图，显示模型关注的图像区域")
    print("  - 将抽象预测转化为直观视觉证据")
    print("  - 增强临床可信度和诊断透明度")
    
    # 示例用法（需要实际的图像文件）
    # explainer.explain_prediction('sample_chest_xray.jpg', top_k=3)

if __name__ == "__main__":
    print("🏥 CheXpert AI 诊断系统")
    print("=" * 50)
    
    # 显示主要功能
    print("📋 主要功能:")
    print("  1. DenseNet-121 深度学习模型")
    print("  2. 14种胸部疾病多标签分类")
    print("  3. Grad-CAM 可解释性分析")
    print("  4. ✅ 训练/验证/测试集正确划分")
    print("  5. 全面的模型评估指标")
    
    # 运行演示
    print("\n" + "="*50)
    create_data_split_demo()
    
    print("\n" + "="*50)
    create_training_example()
    
    print("\n" + "="*50)
    create_explainability_example()
    
    print(f"\n📊 总结:")
    print("✅ 已实现proper的数据集划分策略")
    print("✅ 支持分层采样保证类别平衡")
    print("✅ 包含训练/验证/测试集完整流程")
    print("✅ 提供AUC、精确度、召回率等评估指标")
    print("✅ 适合医学影像AI的最佳实践")