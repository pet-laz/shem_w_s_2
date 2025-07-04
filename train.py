import os
import random
import string
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

# ---------------------- CONFIGURATION ----------------------
DATA_DIR      = 'gen_wave/'    # папка с капчами
IMG_HEIGHT    = 50             # финальная высота после Resize
IMG_WIDTH     = 200            # финальная ширина после Resize
BATCH_SIZE    = 64
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-3
CHARS         = string.ascii_letters + string.digits
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    return batch

# -------------------- DATASET & DATALOADER --------------------
class CaptchaDataset(Dataset):
    def __init__(self, file_list, char_to_idx, transform=None, pad=2):
        self.file_list   = file_list
        self.char_to_idx = char_to_idx
        self.transform   = transform
        self.pad         = pad

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        # Ответ — из имени файла captcha_xxx_<answer>.png
        answer   = os.path.basename(filepath).split('_')[-1].split('.')[0]

        # 1) Загружаем в grayscale и в numpy
        img = Image.open(filepath).convert('L')
        arr = np.array(img)

        # 2) Обрезаем по белым пикселям
        mask = arr > 0
        ys, xs = np.where(mask)
        if xs.size and ys.size:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            x0 = max(0, x0 - self.pad)
            y0 = max(0, y0 - self.pad)
            x1 = min(arr.shape[1], x1 + self.pad)
            y1 = min(arr.shape[0], y1 + self.pad)
            arr = arr[y0:y1, x0:x1]

        # 3) PIL → transform
        img_cropped = Image.fromarray(arr)
        if self.transform:
            img_cropped = self.transform(img_cropped)

        # 4) Кодируем лейбл
        label = torch.tensor([self.char_to_idx[c] for c in answer], dtype=torch.long)
        return img_cropped, label, len(answer)


# Собираем список файлов и делим на трейн/тест
all_files = glob(os.path.join(DATA_DIR, '*.png'))
random.shuffle(all_files)
split = int(0.9 * len(all_files))
train_files, test_files = all_files[:split], all_files[split:]

# Маппинг символов
char_to_idx = {c: i+1 for i, c in enumerate(CHARS)}  # 0 зарезервирован под blank
idx_to_char = {i+1: c for i, c in enumerate(CHARS)}

# Трансформации
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = CaptchaDataset(train_files, char_to_idx, transform)
test_ds  = CaptchaDataset(test_files,  char_to_idx, transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

# ------------------------ MODEL DEFINITION ------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1,   64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256,256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256,512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512,512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(512,512, 2, 1, 0), nn.ReLU()
        )
        # RNN + classifier
        self.lstm = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # x: [B,1,H,W]
        conv = self.cnn(x)                       # [B,512,H',W']
        b, c, h, w = conv.size()
        # «схлопываем» высоту
        conv = F.adaptive_avg_pool2d(conv, (1, w))  # [B,512,1,W']
        conv = conv.squeeze(2)                      # [B,512,W']
        conv = conv.permute(0,2,1)                  # [B,W',512]

        rnn_out, _ = self.lstm(conv)                # [B,W',hidden*2]
        logits     = self.fc(rnn_out)               # [B,W',num_classes]
        return logits.log_softmax(2)


# Instantiate
num_classes = len(CHARS) + 1
model       = CRNN(num_classes).to(DEVICE)
loss_fn     = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ------------------------ TRAIN & EVALUATE ------------------------
def train_epoch(loader):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train batches", ncols=80, leave=False):
        imgs, labels, lengths = zip(*batch)
        imgs = torch.stack(imgs).to(DEVICE)
        targets = torch.cat(labels).to(DEVICE)
        tgt_lengths = torch.tensor(lengths, dtype=torch.long)

        preds = model(imgs)               # [B, W, C]
        preds = preds.permute(1,0,2)      # [T, B, C]
        inp_lengths = torch.full((preds.size(1),), preds.size(0), dtype=torch.long)

        optimizer.zero_grad()
        loss = loss_fn(preds, targets, inp_lengths, tgt_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs, labels, lengths = zip(*batch)
            imgs = torch.stack(imgs).to(DEVICE)
            preds = model(imgs)                   # [B, W, C]
            decoded = []
            seqs = preds.permute(1,0,2).cpu()
            for seq in seqs:
                idxs = torch.argmax(seq, dim=1).numpy().tolist()
                prev = 0; chars = []
                for i in idxs:
                    if i!=prev and i!=0:
                        chars.append(idx_to_char[i])
                    prev = i
                decoded.append(''.join(chars))

            truths = [''.join([idx_to_char[i.item()] for i in lbl]) for lbl in labels]
            for p,t in zip(decoded, truths):
                if p==t: correct+=1
                total+=1
    return correct/total if total>0 else 0.0


# -------------------------- MAIN LOOP --------------------------
if __name__ == '__main__':
    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_epoch(train_loader)
        val_acc    = evaluate(test_loader)
        print(f"  → train loss: {train_loss:.4f}, val acc: {val_acc:.4f}")

    torch.save(model.state_dict(), 'captcha_crnn.pth')
