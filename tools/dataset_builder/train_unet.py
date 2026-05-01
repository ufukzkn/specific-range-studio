"""
U-Net Mimarisi - PyTorch Implementasyonu
=========================================
Bu dosya, U-Net (Convolutional Neural Network) modelinin yapi taslarini icerir.
Kontrol.py tarafindan curve segmentasyonu icin kullanilir.

U-Net Yapisi:
1. Encoder (Daralma Yolu): Goruntu boyutunu kucultup ozellikleri cikarir.
2. Decoder (Genisleme Yolu): Cikarilan ozellikleri orijinal boyuta buyutup segmentasyon yapar.
3. Skip Connections (Atlama Baglantilari): Kaybolan detaylari decoder'a geri ekler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2 Islemi

    Her bir U-Net basamaginda iki kere konvolusyon uygulanir.
    Bu sinif bu tekrarlayan islemi paketler.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # 1. Konvolusyon Blogu
        # - in_channels: Giris kanal sayisi (Orn: Gri resim icin 1)
        # - out_channels: Cikis kanal sayisi (Orn: 64)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 2. Konvolusyon Blogu
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Encoder (Asagi Indirgeme) Blogu

    Yapilan Islem: MaxPool2d (Boyut Kucultme) + DoubleConv (Ozellik Cikarma)
    Goruntu boyutunu yariya indirirken (H/2, W/2), kanal sayisini artirir.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 2x2 Max Pooling -> Boyutu Yariya Indirir
            DoubleConv(in_channels, out_channels) # Ozellikleri cikarir
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder (Yukari Genisletme) Blogu

    Yapilan Islem: Upsampling (Buyutme) + Skip Connection Birlestirme + DoubleConv
    Goruntu boyutunu iki katina cikarir (2H, 2W) ve encoder'dan gelen detaylarla birlestirir.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Upsampling Yontemi Secimi
        # Bilinear: Interpolasyon ile buyutme (Daha hizli, daha az parametre)
        # Transpose Conv: Ogrenilebilir buyutme (Daha yavas ama bazen daha iyi)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: Decoder'dan gelen (alttan yukari cikan) veri
        # x2: Encoder'dan gelen (yan taraftan atlanan) skip connection verisi

        x1 = self.up(x1) # Boyutu buyut

        # Boyut Uyusmazligi Kontrolu (Padding)
        # Bazen pooling islemleri yuzunden boyutlar tam uyusmaz, padding ile duzeltilir.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Skip Connection: Encoder'dan gelen x2 ile Decoder'dan gelen x1'i birlestir
        x = torch.cat([x2, x1], dim=1)

        # Birlestirilmis veriyi isle
        return self.conv(x)


class OutConv(nn.Module):
    """
    Cikis Katmani (Output Convolution)

    Cikarilan ozellikleri sonuca (segmentasyon maskesine) donusturur.
    Kanal sayisini sinif sayisina (binary ise 1) indirir.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Tam U-Net Modeli Mimarisi
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- ENCODER (Giris & Daralma) ---
        self.inc = DoubleConv(n_channels, 64)       # Giris: 1 -> 64
        self.down1 = Down(64, 128)                  # 64 -> 128 (Boyut / 2)
        self.down2 = Down(128, 256)                 # 128 -> 256 (Boyut / 4)
        self.down3 = Down(256, 512)                 # 256 -> 512 (Boyut / 8)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)      # En Alt Katman: 512 -> 1024 (Boyut / 16)

        # --- DECODER (Cikis & Genisletme) ---
        self.up1 = Up(1024, 512 // factor, bilinear) # 1024 -> 512 (Boyut x 2)
        self.up2 = Up(512, 256 // factor, bilinear)  # 512 -> 256 (Boyut x 4)
        self.up3 = Up(256, 128 // factor, bilinear)  # 256 -> 128 (Boyut x 8)
        self.up4 = Up(128, 64, bilinear)             # 128 -> 64  (Boyut x 16 -> Orijinal Boyut)

        # --- CIKIS ---
        self.outc = OutConv(64, n_classes)           # 64 -> Cikis Sinifi (1)

    def forward(self, x):
        """
        Modelin Ileri Yayilimi (Forward Pass)
        Veri modelin icinde nasil akar?
        """
        # Encoder Asamalari (Veriyi sakla x1, x2... skip connection icin)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder Asamalari (Saklanan verilerle birlestirerek yukari cik)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Sonuc Maskesi
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Hafiza tasarrufu icin gradient checkpointing kullan (Opsiyonel)"""
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# ==================================================================================================
# DATASET & DATALOADER
# ==================================================================================================
import os
import cv2
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

class CurveSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), split='all', split_file=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.split = split

        # Dosya listesini olustur
        self.images = []

        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                data = json.load(f)
                # JSON yapisina gore dosya isimlerini al
                if isinstance(data, list):
                    self.images = [item['file_name'] for item in data]
                elif isinstance(data, dict) and 'images' in data:
                    self.images = [item['file_name'] for item in data['images']]
        else:
            # Split dosyasi yoksa klasordeki tum png'leri al
            if os.path.exists(image_dir):
                self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            else:
                print(f"[UYARI] Goruntu klasoru bulunamadi: {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Maske ismini tahmin et (img_X -> mask_X)
        mask_name = img_name.replace('img_', 'mask_') if 'img_' in img_name else img_name
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Goruntu Yukle (Grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Hata durumunda bos dondur (veya hata firlat)
            print(f"[HATA] Goruntu okunamadi: {img_path}")
            return torch.zeros((1, *self.image_size)), torch.zeros((1, *self.image_size))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
             # Maske yoksa (orn: test seti), bos maske dondur
            mask = np.zeros_like(image)

        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)

        # Normalize & Tensor
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32) # Binary mask

        # Channel boyutu ekle (H, W) -> (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# ==================================================================================================
# TRAINER
# ==================================================================================================
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

class CurveSegmentationTrainer:
    def __init__(self, device='cuda', checkpoint_dir='checkpoints'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
        print(f"[Trainer] Device: {self.device}")

    def create_model(self, in_channels=1, out_channels=1, features=64):
        model = UNet(n_channels=in_channels, n_classes=out_channels, bilinear=True)
        return model.to(self.device)

    def _dice_loss(self, pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def _iou_score(self, pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def train_epoch(self, model, loader, optimizer):
        model.train()
        epoch_loss = 0
        epoch_iou = 0

        pbar = tqdm(loader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = self._dice_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_iou += self._iou_score(outputs, masks).item()

            pbar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(loader), epoch_iou / len(loader)

    def validate(self, model, loader):
        model.eval()
        epoch_loss = 0
        epoch_iou = 0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = model(images)
                loss = self._dice_loss(outputs, masks)

                epoch_loss += loss.item()
                epoch_iou += self._iou_score(outputs, masks).item()

        return epoch_loss / len(loader), epoch_iou / len(loader)

    def fit(self, model, train_loader, val_loader, epochs=50, learning_rate=1e-3, patience=10):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n[Basliyor] Toplam Epoch: {epochs}")

        for epoch in range(epochs):
            train_loss, train_iou = self.train_epoch(model, train_loader, optimizer)
            val_loss, val_iou = self.validate(model, val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                  f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, 'best_model.pth'))
                print(f"  [✓] Model kaydedildi (Loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    print(f"\n[Erken Durdurma] {patience} epoch boyunca iyilesme olmadi. Egitim kesiliyor.")
                    break

# ==================================================================================================
# MAIN EXECUTION
# ==================================================================================================
if __name__ == '__main__':
    import argparse
    import sys

    # Argumanlari tanimla
    parser = argparse.ArgumentParser(description="U-Net Model Egitimi")
    parser.add_argument('--dataset', type=str, default='dataset_production', help='Dataset klasoru')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch sayisi')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--image_size', type=int, default=256, help='Giris resim boyutu')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker sayisi')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("U-NET EGITIM MODULU")
    print(f"{'='*70}")

    # Dataset Yollarini Kontrol Et (Roboflow veya Klasik Yapı)
    # Yeni yapı: dataset/splits/train
    # Eski yapı: dataset/train

    splits_parent = os.path.join(args.dataset, 'splits')

    # Pratiklik: Hangi yol varsa onu seç
    if os.path.exists(os.path.join(splits_parent, 'train')):
        train_dir = os.path.join(splits_parent, 'train')
        valid_dir = os.path.join(splits_parent, 'valid')
        test_dir = os.path.join(splits_parent, 'test')
    else:
        train_dir = os.path.join(args.dataset, 'train')
        valid_dir = os.path.join(args.dataset, 'valid')
        test_dir = os.path.join(args.dataset, 'test')

    # Roboflow stili (train/valid/test klasörleri var mı?)
    is_roboflow_style = os.path.exists(train_dir) and os.path.exists(valid_dir)

    if is_roboflow_style:
        print("[INFO] Roboflow stili üçlü klasör yapısı tespit edildi.")
        # Görseller kök dizinde mi yoksa 'images' alt klasöründe mi?
        train_img_dir = os.path.join(train_dir, 'images') if os.path.exists(os.path.join(train_dir, 'images')) else train_dir
        train_mask_dir = os.path.join(train_dir, 'masks')
        # .coco veya .coco.json dosyalarını ara
        train_ann = os.path.join(train_dir, '_annotations.coco')
        if not os.path.exists(train_ann):
            train_ann = os.path.join(train_dir, '_annotations.coco.json')

        valid_img_dir = os.path.join(valid_dir, 'images') if os.path.exists(os.path.join(valid_dir, 'images')) else valid_dir
        valid_mask_dir = os.path.join(valid_dir, 'masks')
        valid_ann = os.path.join(valid_dir, '_annotations.coco')
        if not os.path.exists(valid_ann):
            valid_ann = os.path.join(valid_dir, '_annotations.coco.json')

        train_ds = CurveSegmentationDataset(train_img_dir, train_mask_dir, (args.image_size, args.image_size),
                                          split_file=train_ann if os.path.exists(train_ann) else None)
        val_ds = CurveSegmentationDataset(valid_img_dir, valid_mask_dir, (args.image_size, args.image_size),
                                        split_file=valid_ann if os.path.exists(valid_ann) else None)
    else:
        # Klasik yapı (images/masks klasörleri ana dizinde)
        images_dir = os.path.join(args.dataset, 'images')
        masks_dir = os.path.join(args.dataset, 'masks')
        splits_dir = os.path.join(args.dataset, 'splits')

        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"[HATA] Dataset klasor yapisi gecersiz: {args.dataset}")
            print("Beklenen yapi: dataset/train veya dataset/images")
            sys.exit(1)

        print("[INFO] Klasik dataset yapısı kullanılıyor.")
        if os.path.exists(os.path.join(args.dataset, 'splits', 'train.json')):
            train_ds = CurveSegmentationDataset(images_dir, masks_dir, (args.image_size, args.image_size),
                                              split_file=os.path.join(splits_dir, 'train.json'))
            val_ds = CurveSegmentationDataset(images_dir, masks_dir, (args.image_size, args.image_size),
                                            split_file=os.path.join(splits_dir, 'val.json'))
        else:
            full_ds = CurveSegmentationDataset(images_dir, masks_dir, (args.image_size, args.image_size))
            train_size = int(0.8 * len(full_ds))
            val_size = len(full_ds) - train_size
            train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    print(f"  - Train Set: {len(train_ds)}")
    print(f"  - Val Set: {len(val_ds)}")

    # Dataloader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Trainer
    print("[2/4] Trainer hazirlaniyor...")
    trainer = CurveSegmentationTrainer(device=args.device)
    model = trainer.create_model()

    # Egitim
    print("[3/4] Egitim basliyor...")
    trainer.fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience
    )

    print("\n[4/4] Islem tamamlandi!")
    print(f"Model kaydi: checkpoints/best_model.pth")
