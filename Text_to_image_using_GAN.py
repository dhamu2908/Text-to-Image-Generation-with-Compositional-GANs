import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import clip
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm

# ======================
# Configuration
# ======================
class Config:
    # Model parameters
    latent_dim = 256
    text_embed_dim = 256
    img_size = 256
    channels = 3
    ngf = 64  # Generator feature maps
    ndf = 64  # Discriminator feature maps
    
    # Training parameters
    batch_size = 32
    epochs = 200
    lr = 0.0002
    beta1 = 0.5
    clip_weight = 0.5
    
    # Dataset paths
    coco_path = "coco_dataset/"
    annotations_file = "annotations/captions_train2017.json"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Dataset Preparation
# ======================
class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load annotations
        with open(ann_file) as f:
            annotations = json.load(f)
        
        # Create image-caption pairs
        self.data = []
        for ann in annotations['annotations']:
            img_id = ann['image_id']
            img_path = os.path.join(root_dir, f"{img_id:012d}.jpg")
            if os.path.exists(img_path):
                self.data.append({
                    'image_path': img_path,
                    'caption': ann['caption']
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption = item['caption']
        tokens = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=64, 
            return_tensors='pt'
        )
        
        return image, tokens

# ======================
# Model Architectures
# ======================
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Sequential(
            nn.Linear(768, Config.text_embed_dim),
            nn.BatchNorm1d(Config.text_embed_dim),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        return self.fc(pooled_output)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial projection
        self.projection = nn.Sequential(
            nn.Linear(Config.latent_dim + Config.text_embed_dim, 8 * 8 * Config.ngf * 8),
            nn.BatchNorm1d(8 * 8 * Config.ngf * 8),
            nn.ReLU()
        )
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            self._make_up_block(Config.ngf * 8, Config.ngf * 8),  # 8x8 -> 16x16
            self._make_up_block(Config.ngf * 8, Config.ngf * 4),  # 16x16 -> 32x32
            self._make_up_block(Config.ngf * 4, Config.ngf * 2),  # 32x32 -> 64x64
            self._make_up_block(Config.ngf * 2, Config.ngf),      # 64x64 -> 128x128
            self._make_up_block(Config.ngf, Config.ngf // 2)      # 128x128 -> 256x256
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=Config.ngf // 2, num_heads=8)
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(Config.ngf // 2, Config.channels, 3, padding=1),
            nn.Tanh()
        )
    
    def _make_up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, noise, text_embed):
        # Concatenate noise and text embedding
        x = torch.cat([noise, text_embed], dim=1)
        x = self.projection(x)
        x = x.view(-1, Config.ngf * 8, 8, 8)
        
        # Upsampling
        for block in self.upsample_blocks:
            x = block(x)
        
        # Attention between image features and text
        batch, channels, height, width = x.shape
        x_flat = x.view(batch, channels, -1).permute(2, 0, 1)  # (H*W, B, C)
        text_embed = text_embed.unsqueeze(0).repeat(height * width, 1, 1)
        
        attn_output, _ = self.attention(x_flat, text_embed, text_embed)
        attn_output = attn_output.permute(1, 2, 0).view(batch, channels, height, width)
        
        # Residual connection
        x = x + attn_output
        
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Image processing pathway
        self.img_net = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(Config.channels, Config.ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # 64 x 128 x 128
            nn.Conv2d(Config.ndf, Config.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(Config.ndf * 2),
            nn.LeakyReLU(0.2),
            # 128 x 64 x 64
            nn.Conv2d(Config.ndf * 2, Config.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(Config.ndf * 4),
            nn.LeakyReLU(0.2),
            # 256 x 32 x 32
            nn.Conv2d(Config.ndf * 4, Config.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(Config.ndf * 8),
            nn.LeakyReLU(0.2),
            # 512 x 16 x 16
        )
        
        # Text processing pathway
        self.text_net = nn.Sequential(
            nn.Linear(Config.text_embed_dim, Config.ndf * 8),
            nn.BatchNorm1d(Config.ndf * 8),
            nn.LeakyReLU(0.2)
        )
        
        # Joint processing
        self.joint_net = nn.Sequential(
            nn.Linear(Config.ndf * 8 * 16 * 16 + Config.ndf * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, text_embed):
        # Process image
        img_feat = self.img_net(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        
        # Process text
        text_feat = self.text_net(text_embed)
        
        # Joint processing
        joint = torch.cat([img_feat, text_feat], dim=1)
        return self.joint_net(joint)

# ======================
# Training Setup
# ======================
def initialize_models():
    # Models
    text_encoder = TextEncoder().to(Config.device)
    generator = Generator().to(Config.device)
    discriminator = Discriminator().to(Config.device)
    
    # CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=Config.device)
    
    # Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.beta1, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=Config.lr, betas=(Config.beta1, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    return text_encoder, generator, discriminator, clip_model, opt_G, opt_D, criterion

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = CocoDataset(
        root_dir=os.path.join(Config.coco_path, "train2017"),
        ann_file=os.path.join(Config.coco_path, Config.annotations_file),
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

# ======================
# Training Loop
# ======================
def train():
    # Initialize everything
    text_encoder, G, D, clip_model, opt_G, opt_D, criterion = initialize_models()
    dataloader = get_dataloader()
    
    # Freeze text encoder and CLIP
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # Training loop
    for epoch in range(Config.epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.epochs}")
        
        for real_imgs, captions in progress_bar:
            # Move data to device
            real_imgs = real_imgs.to(Config.device)
            input_ids = captions['input_ids'].squeeze(1).to(Config.device)
            attention_mask = captions['attention_mask'].squeeze(1).to(Config.device)
            
            # Get text embeddings
            with torch.no_grad():
                text_emb = text_encoder(input_ids, attention_mask)
                clip_text_features = clip_model.encode_text(
                    clip.tokenize([cap for cap in captions['caption']], truncate=True).to(Config.device)
                )
            
            # Generate fake images
            noise = torch.randn(real_imgs.size(0), Config.latent_dim).to(Config.device)
            fake_imgs = G(noise, text_emb)
            
            # Train Discriminator
            D.zero_grad()
            
            # Real images
            real_labels = torch.ones(real_imgs.size(0), 1).to(Config.device)
            real_output = D(real_imgs, text_emb)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            fake_labels = torch.zeros(real_imgs.size(0), 1).to(Config.device)
            fake_output = D(fake_imgs.detach(), text_emb)
            d_loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_D.step()
            
            # Train Generator
            G.zero_grad()
            
            # Adversarial loss
            g_output = D(fake_imgs, text_emb)
            g_loss_adv = criterion(g_output, real_labels)
            
            # CLIP loss for semantic alignment
            clip_img_features = clip_model.encode_image(
                (fake_imgs + 1) / 2  # CLIP expects [0,1] range
            )
            g_loss_clip = 1 - F.cosine_similarity(clip_img_features, clip_text_features).mean()
            
            # Total generator loss
            g_loss = g_loss_adv + Config.clip_weight * g_loss_clip
            g_loss.backward()
            opt_G.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': d_loss.item(),
                'G_loss': g_loss.item(),
                'CLIP_loss': g_loss_clip.item()
            })
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save({
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'epoch': epoch
            }, f"checkpoint_epoch_{epoch+1}.pth")

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    train()