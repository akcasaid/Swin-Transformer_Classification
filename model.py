import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=96):
        super(PatchEmbed, self).__init__()
        # Görüntüleri küçük parçalara bölmek için konvolüsyon katmanı
        # patch_size, görüntüden çıkarılacak parçaların boyutunu belirtir
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Görüntüleri parçalara ayırır ve bu parçaları gömme vektörlerine dönüştürür
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(MLP, self).__init__()
        # İki tam bağlantılı katman ve aktivasyon fonksiyonu içerir
        # İlk katman, giriş özelliklerini gizli özelliklere dönüştürür
        # İkinci katman, gizli özellikleri tekrar giriş özelliklerine dönüştürür
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()  # GELU aktivasyon fonksiyonu kullanılır

    def forward(self, x):
        # MLP'den geçiş
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super(SwinBlock, self).__init__()
        # Multi-head dikkat mekanizması
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        # MLP, her SwinBlock'ta özelliği dönüştürmek için kullanılır
        self.mlp = MLP(dim, dim * 4)
        self.norm1 = nn.LayerNorm(dim)  # Dikkat sonrası normalizasyon
        self.norm2 = nn.LayerNorm(dim)  # MLP sonrası normalizasyon
        self.window_size = window_size
        self.shift_size = shift_size

    def forward(self, x):
        B, N, C = x.shape  # B: batch size, N: patch sayısı, C: özellik sayısı
        x = self.norm1(x)  # İlk normalizasyon
        x = x.permute(1, 0, 2)  # [N, B, C] şekline dönüştür
        x, _ = self.attn(x, x, x)  # Multi-head dikkat
        x = x.permute(1, 0, 2)  # [B, N, C] şekline dönüştür
        x = x + self.norm2(self.mlp(x))  # MLP ve normalizasyon sonrası ekleme
        return x

class SwinTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):  # 4 sınıf için ayarlandı
        super(SwinTransformer, self).__init__()
        self.patch_embed = PatchEmbed(in_channels=in_channels)  # Görüntüleri parçalara böler
        self.swin_block = SwinBlock(dim=96, num_heads=3)  # Swin Transformer bloğu
        self.fc = nn.Linear(96, num_classes)  # Son katman, sınıflandırma için

    def forward(self, x):
        x = self.patch_embed(x)  # Patch'lere dönüştür
        x = x.flatten(2).transpose(1, 2)  # [B, N, C] şekline dönüştür
        x = self.swin_block(x)  # SwinBlock'tan geçir
        x = x.mean(dim=1)  # Orta havuzlama
        x = self.fc(x)  # Sınıflandırma çıkışı
        return x

# Modeli oluşturma
model = SwinTransformer(num_classes=4)  # 4 sınıflı problemi çözmek için model
print(model)
