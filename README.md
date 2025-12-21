WGAN-based Anomaly Detection for MVTec LOCO

Ovaj projekat implementira poboljšanu f-AnoGAN arhitekturu za nenadgledanu detekciju anomalija u industrijskim slikama, sa posebnim fokusom na logičke (LOCO) anomalije. Metoda kombinuje patch-based analizu, WGAN-GP generativni model, enkoder za rekonstrukciju, kao i globalni model za prostorne odnose.

Struktura projekta
.
├── WGAN_v2.py          # Glavna skripta
├── cfg.py              # Konfiguracioni fajl
├── plots.py            # Funkcije za evaluaciju i vizualizaciju
├── outputs/            # Generisani uzorci, rekonstrukcije i grafici
├── checkpoints/        # Sačuvani modeli
└── README.md

Instalacija zavisnosti:
pip install torch torchvision numpy matplotlib seaborn pandas scipy tqdm pillow

Skup podataka

Koristi se MVTec LOCO AD dataset.
Struktura dataset-a mora biti sljedeća:

MVTec_LOCO/
└── category_name/
    ├── train/
    │   └── good/
    ├── validation/
    │   └── good/
    └── test/
        ├── good/
        └── logical_anomalies/

Primjer kategorije: breakfast_box, pushpins, juice_bottle.

Faze rada:
Skripta se pokreće u tri glavne faze.

1️. Treniranje WGAN-GP modela
Poziv iz terminala:
python WGAN_v2.py \
  --data_root /putanja/do/MVTec_LOCO \
  --category breakfast_box \
  --phase train_gan \
  --epochs 450 --batch 32

Rezultat:

sačuvani GAN modeli (checkpoints/gan_latest.pth)
generisani uzorci u outputs/

2️. Treniranje enkodera
Poziv iz terminala: 

python WGAN_v2.py \
  --data_root /putanja/do/MVTec_LOCO \
  --category breakfast_box \
  --phase train_encoder \
  --ckpt checkpoints/gan_latest.pth \
  --epochs 100 --batch 32

Rezultat:
sačuvan enkoder (checkpoints/encoder_latest.pth)
rekonstrukcije i PSNR mjere u outputs/

3️. Evaluacija i detekcija anomalija

U fazi evaluacije računaju se:
patch-level anomalijski skorovi,
globalni anomalijski skor (opciono),
konačna odluka po slici.

Bez globalnog modela:

python WGAN_v2.py \
  --data_root /putanja/do/MVTec_LOCO \
  --category breakfast_box \
  --phase eval \
  --ckpt checkpoints/gan_latest.pth \
  --enc_ckpt checkpoints/encoder_latest.pth \
   --global_model none


Sa globalnim modelom:

python WGAN_v2.py \
  --data_root /putanja/do/MVTec_LOCO \
  --category breakfast_box \
  --phase eval \
  --ckpt checkpoints/gan_latest.pth \
  --enc_ckpt checkpoints/encoder_latest.pth \
  --global_model spatial_transformer

Globalni model (Spatial Transformer)

Opcioni globalni model analizira cijelu sliku i uči prostorne odnose između regiona.
Koristi:

CNN ekstrakciju feature-a,
prostornu attention mapu,
globalni prosjek (global average pooling).
Globalni anomalijski skor se računa kao kombinacija:
kosinusne udaljenosti globalnih feature-a,
razlike attention mapa između originala i rekonstrukcije.

Izlaz i evaluacija

Tokom evaluacije se generišu:

ROC i Precision–Recall krive
AUC i AP metričke vrijednosti
optimalni prag anomalije
heatmap-e anomalnih patch-eva
CSV fajl sa rezultatima

Sve se čuva u:

outputs/evaluation/<category>/<global_model>/
