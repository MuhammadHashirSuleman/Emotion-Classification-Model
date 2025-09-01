import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import yaml
import time
from tqdm import tqdm
import os

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedEmotionTrainer:
    def __init__(self, config_path="config/params.yaml"):
        self.config_path = config_path
        self.load_config()
        self.setup_device()
        self.load_data()
        self.calculate_class_weights()
        self.setup_model()
        self.setup_optimizer()
        
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config['model_name']
        self.num_labels = self.config['num_labels']
        self.batch_size = self.config['training_args']['per_device_train_batch_size']
        self.learning_rate = self.config['training_args']['learning_rate']
        self.num_epochs = self.config['training_args']['num_train_epochs']
        self.accumulation_steps = self.config['training_args'].get('gradient_accumulation_steps', 4)
        
    def setup_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_data(self):
        from data_processing import load_and_process_data
        self.dataset, self.emotion_labels = load_and_process_data(self.config_path)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.dataset['train'], 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        self.val_loader = DataLoader(
            self.dataset['validation'], 
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
        
    def calculate_class_weights(self):
        """Calculate class weights based on label frequency"""
        # Extract labels from the dataset and convert to tensor
        train_labels = [example['labels'] for example in self.dataset['train']]
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        
        class_counts = torch.sum(train_labels_tensor, dim=0)
        
        # Avoid division by zero and extreme weights
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_labels
        self.class_weights = class_weights.to(self.device)
        print("Class weights:", class_weights.cpu().numpy())
        
    def collate_fn(self, batch):
        """Custom collation function for multi-label classification"""
        input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.float)
        return {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'labels': labels.to(self.device)
        }
        
    def setup_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification"
        ).to(self.device)
        
        # Use Focal Loss with class weighting
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Use Cosine Annealing instead of Linear
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
    def compute_metrics(self, predictions, labels, thresholds=None):
        """Compute advanced metrics for multi-label classification"""
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(predictions)
        
        if thresholds is not None:
            # Use custom thresholds if provided
            y_pred = (probs >= thresholds.to(self.device)).int().cpu().numpy()
        else:
            # Default 0.5 threshold
            y_pred = (probs >= 0.5).int().cpu().numpy()
            
        y_true = labels.int().cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        }
        return metrics, probs
        
    def optimize_thresholds(self, val_loader):
        """Find optimal thresholds per class using validation set"""
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Optimizing thresholds"):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                probs = torch.sigmoid(outputs.logits)
                all_probs.append(probs)
                all_labels.append(batch['labels'])
        
        all_probs = torch.cat(all_probs).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        optimal_thresholds = []
        for i in range(self.num_labels):
            precision, recall, thresholds = precision_recall_curve(
                all_labels[:, i], all_probs[:, i]
            )
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.nanargmax(f1_scores)
            optimal_thresholds.append(thresholds[best_idx] if best_idx < len(thresholds) else 0.5)
        
        return torch.tensor(optimal_thresholds)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        self.optimizer.zero_grad()
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Calculate loss with class weighting
            loss = self.criterion(outputs.logits, batch['labels'])
            loss = loss / self.accumulation_steps  # Normalize for accumulation
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            all_predictions.append(outputs.logits.detach())
            all_labels.append(batch['labels'].detach())
            
            progress_bar.set_postfix({'loss': loss.item() * self.accumulation_steps})
            
        # Handle remaining batches
        if len(self.train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.scheduler.step()
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics, _ = self.compute_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
        
    def validate_epoch(self, thresholds=None):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch in progress_bar:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                loss = self.criterion(outputs.logits, batch['labels'])
                total_loss += loss.item()
                
                all_predictions.append(outputs.logits)
                all_labels.append(batch['labels'])
                
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics, probs = self.compute_metrics(all_predictions, all_labels, thresholds)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics, probs, all_labels
        
    def train(self):
        print("Starting advanced training with Focal Loss and class weighting...")
        best_f1_macro = 0
        best_model_state = None
        self.optimal_thresholds = None
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time
            
            # Validate
            val_metrics, val_probs, val_labels = self.validate_epoch(self.optimal_thresholds)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train F1 Micro: {train_metrics['f1_micro']:.4f} | "
                  f"Time: {train_time:.2f}s")
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val F1 Micro: {val_metrics['f1_micro']:.4f} | "
                  f"Val F1 Macro: {val_metrics['f1_macro']:.4f}")
            
            # Save best model based on Macro F1 (better for imbalance)
            if val_metrics['f1_macro'] > best_f1_macro:
                best_f1_macro = val_metrics['f1_macro']
                best_model_state = self.model.state_dict().copy()
                
                # Optimize thresholds on best validation performance
                print("🔍 Optimizing thresholds...")
                self.optimal_thresholds = self.optimize_thresholds(self.val_loader)
                print(f"Optimal thresholds: {self.optimal_thresholds.numpy()}")
                
                # Save best model
                torch.save(best_model_state, "models/saved_model/pytorch_model.bin")
                print("💾 Saved best model with optimized thresholds!")
                
        print("\nTraining completed!")
        
        # Load best model for final save
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
    def save_model(self):
        # Save final model with all components
        self.model.save_pretrained("models/saved_model")
        
        # Save optimal thresholds
        if self.optimal_thresholds is not None:
            torch.save(self.optimal_thresholds, "models/saved_model/optimal_thresholds.pt")
        
        print("Model and thresholds saved to models/saved_model")

if __name__ == "__main__":
    trainer = AdvancedEmotionTrainer()
    trainer.train()
    trainer.save_model()



