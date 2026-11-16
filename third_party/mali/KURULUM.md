# Ortam Kurulumu — MTAH 2027

Bu doküman, 5 kişilik takımın her üyesinin **aynı Python ortamını tek komutla** kurması ve veri dosyalarını resmi reponun beklediği düzene yerleştirmesi içindir. Kurulum Windows, macOS ve Linux'ta aynı paket sürümlerini kurar (`kurulum/requirements.txt` platformlar arası kilit dosyası).

> 21 Ekim 2026'da bu makinede (Windows 11, RTX 5070 Ti Laptop, Python 3.11.9) sıfırdan çalıştırıldı. Sonuç: **32 kontrol, 0 hata** (`--video` ile 34 kontrol, 0 hata, 2 bilinen uyarı).

---

## 1. Ön koşullar (bir kez)

| Araç | Neden | Kurulum |
|---|---|---|
| **git** | Resmi repoyu klonlamak | Windows: `winget install Git.Git` · macOS: `xcode-select --install` · Linux: `sudo apt install git` |
| **uv** | Python 3.11'i ve paketleri kilit dosyasına göre kurmak (Python yoksa kendisi indirir) | Aşağıdaki komutlar |

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatif (her platform): `pip install uv`

**ffmpeg'i ayrıca kurmanıza gerek yok.** Ortam, `imageio-ffmpeg` paketiyle gelen hazır ffmpeg'i `.venv` içine `ffmpeg` komutu olarak yerleştirir. Video meta verisi için PyAV (`av`) kullanılır.

---

## 2. Klasör düzeni

```
barca/                                ← çalışma klasörü (adı önemli değil)
├─ kurulum/                           ← bu kurulum scriptleri
│  ├─ setup_env.ps1                   Windows tek komut kurulum
│  ├─ setup_env.sh                    macOS / Linux tek komut kurulum
│  ├─ requirements.in                 doğrudan bağımlılıklar (elle düzenlenen)
│  ├─ requirements.txt                kilit dosyası (uv ile üretilir, elle düzenlenmez)
│  ├─ organize_data.py                indirilen veriyi doğru klasörlere yerleştirir
│  └─ check_env.py                    ortamı + veriyi + hizalamayı + puanlayıcıyı doğrular
├─ more-than-a-hack-2027/             ← resmi repo (git clone)
│  └─ data/                           ← veri burada (gitignore'lu)
│     ├─ tactical/   BAR-ATH_tactical.mp4, _atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
│     └─ broadcast/  BAR-ATH_broadcast.mp4, _atd.xml, _atd.txt, _homography.csv
├─ arastirma_sonuclari/               ← Gemini Deep Research raporları
└─ .venv/                             ← kurulumun oluşturduğu sanal ortam (paylaşılmaz, commit edilmez)
```

Resmi repo henüz yoksa `barca/` içinde:

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
```

---

## 3. Veriyi yerleştirme (bir kez)

İndirdiğiniz veri paketini açtığınız klasörü verin. Script dosyaları adlarına göre bulur, `data/tactical` ve `data/broadcast` altına taşır ve videoları `games.py`'nin beklediği adlara çevirir (ör. `BAR-ATH_tactical-001.mp4` → `BAR-ATH_tactical.mp4`). Sadece standart kütüphane kullanır, sistemdeki herhangi bir Python 3 ile çalışır.

```bash
python kurulum/organize_data.py <indirme_klasörü> --dry-run
```

```bash
python kurulum/organize_data.py <indirme_klasörü>
```

- Orijinaller yerinde kalsın istiyorsanız `--copy` ekleyin (19 GB ek disk ister).
- Veriyi başka bir diskte tutmak isterseniz `MTAH_DATA_DIR` ortam değişkenini o klasöre ayarlayın. Hem script hem resmi repo bu değişkeni kullanır.

---

## 4. Ortamı kurma — tek komut

Windows (PowerShell, `barca/` klasöründe):

```powershell
powershell -ExecutionPolicy Bypass -File kurulum\setup_env.ps1
```

macOS / Linux:

```bash
bash kurulum/setup_env.sh
```

Script şunları yapar:
1. `.venv` yoksa Python 3.11 ile oluşturur (Python 3.11 kurulu değilse uv indirir).
2. `uv pip sync` ile **tam olarak** kilit dosyasındaki paketleri kurar. Fazlalıkları da kaldırır, böylece herkesin ortamı birebir aynı olur.
3. ffmpeg'i `.venv` içine yerleştirir.
4. `check_env.py` ile doğrulama yapar. Beklenen son satır: `32 checks, 0 failed.`

Video kare sayılarını da doğrulamak için `-Video` / `--video` (birkaç saniye) veya `-Decode` / `--decode` (tactical videonun tamamı decode edilir, birkaç dakika) ekleyin.

Ortamı etkinleştirme:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

### `check_env.py` neleri doğrular?

| Bölüm | Kontrol |
|---|---|
| Environment | Python 3.11, 12 paket import + sürüm, `ffmpeg` komutu |
| Data layout | 9 veri dosyasının beklenen yerde olması |
| Alignment | Resmi starter kontrolleri: ATD 199.319 / 164.488 kayıt ve `frame_id` 1'den ardışık; homografi satır sayıları; Smart Tagging zaman tabanı (başlama vuruşları) |
| kloppy | EPTS tracking ve SportsCode dosyalarının kloppy 3.18.0 + pandas 3 ile okunması |
| Scorer | Resmi puanlayıcının yayınlanmış public skorları birebir üretmesi (smooth50 0,3256; smooth125 0,4242) |
| Video (`--video`) | mp4 başlığındaki kare sayısı ve demux edilen paket sayısı. Tactical = 199.319 olmalı. Broadcast'te bilinen 1 kare eksiklik (164.487) `[WARN]` olarak gösterilir, hata sayılmaz |
| Video (`--decode`) | ffmpeg ile tactical videonun tüm karelerinin decode edilip sayılması |

---

## 5. Paket ekleme / güncelleme (kurallar)

**Kimse ortama elle `pip install` yapmaz.** Aksi halde ortamlar ayrışır ve organizatörlerin istediği "birebir aynı dosya" yeniden üretimi riske girer.

1. `kurulum/requirements.in` dosyasına paketi ekleyin (gerekirse sürümüyle).
2. Kilidi yeniden üretin:

```bash
uv pip compile kurulum/requirements.in --universal --python-version 3.11 -o kurulum/requirements.txt
```

3. Herkes kurulum scriptini tekrar çalıştırır (sadece farkları kurar).
4. `requirements.in` ve `requirements.txt` birlikte commit edilir.

Modelleme kütüphaneleri (PyTorch, LightGBM vb.) araştırma raporları geldikten sonra bu yolla eklenecek. Not: RTX 50 serisi (Blackwell) GPU'lar CUDA 12.8+ derlemeli PyTorch ister. Bu, kilit dosyasına özel index ile eklenecek.

---

## 6. Yeniden üretilebilirlik kuralları (şimdiden geçerli)

- **CSV'leri her zaman `\n` satır sonuyla yazın.** Kurulum testinde resmi `baseline.py`'yi bu makinede çalıştırdık: içerik repodaki dosyalarla birebir aynı çıktı, ama Windows `\r\n` yazdığı için byte düzeyinde farklıydı. Organizatörler "aynı dosya" kontrolünü başka bir işletim sisteminde yapabilir. Python'da `open(path, "w", newline="\n")` veya `df.to_csv(path, index=False, lineterminator="\n")` kullanın.
- Rastgelelik içeren her adımda seed sabitlenir. GPU'da deterministik ayarlar kullanılır.
- Veri yolu asla kodda sabit yazılmaz: `games.asset_paths()` veya `MTAH_DATA_DIR` kullanılır.
- Resmi puanlayıcıyı repo içine çıktı yazdırmadan çalıştırın: `python more-than-a-hack-2027/evaluation/run_evaluation.py --out <repo_dışı_klasör>`.
- `.venv/`, `data/`, videolar, GT dosyaları ve gönderim CSV'leri takımın public reposuna **asla** girmez.

---

## 7. Sorun giderme

| Belirti | Çözüm |
|---|---|
| `uv bulunamadi` | Bölüm 1'deki uv kurulumunu yapın, terminali kapatıp açın. |
| PowerShell "running scripts is disabled" | Komutu `powershell -ExecutionPolicy Bypass -File ...` şeklinde çalıştırın (yukarıdaki gibi). |
| `match data not found` / Data layout FAIL | `organize_data.py` ile veriyi yerleştirin veya `MTAH_DATA_DIR`'ı doğru klasöre ayarlayın. |
| Paket kurulumu ağ hatası | Tekrar çalıştırın. `uv` önbelleği kaldığı yerden devam eder. |
| Büyük wheel (ör. 40 MB `llvmlite`) indirmesi takılıyor | Wheel'i `curl -C -` ile kaldığı yerden devam ederek indirin, sha256'sını `https://pypi.org/pypi/<paket>/<sürüm>/json` ile karşılaştırın, `uv pip install --python .venv\Scripts\python.exe <wheel>` ile kurun, sonra kurulum scriptini tekrar çalıştırın. numba yokken decoder otomatik olarak aynı sonucu veren numpy yoluna düşer. |
| Sistem Python'u açılışta `orange3...pth` hatası yazıyor | `.venv` içinde görülmez. Sadece sistem Python'unu etkiler, zararsız. |
| Jupyter yanlış Python'u kullanıyor | Önce ortamı etkinleştirin, sonra `jupyter notebook` çalıştırın. |

---

## 8. Bu makinede doğrulanan değerler (21 Ekim 2026)

| Kontrol | Sonuç |
|---|---|
| Python | 3.11.9 |
| Paketler | numpy 2.4.6 · pandas 3.0.6 · scipy 1.17.1 · scikit-learn 1.9.1 · pyarrow 26.0.0 · kloppy 3.18.0 · matplotlib 3.11.2 · notebook 7.6.3 · av 18.1.0 · imageio-ffmpeg 0.6.0 (ffmpeg 7.1) |
| ATD hizalama | tactical 199.319 kayıt, broadcast 164.488 kayıt, `frame_id` 1'den ardışık |
| Homografi | tactical 199.319, broadcast 164.488 satır |
| Smart Tagging zaman tabanı | 2 başlama vuruşu etiketi gerçek başlama vuruşlarını kapsıyor |
| Resmi puanlayıcı | smooth50 0,3256 · smooth125 0,4242 (yayınlanan değerlerle aynı) |
| `baseline.py` yeniden üretimi | 4 dosyanın içeriği repodakilerle birebir aynı (sadece satır sonu farkı, bkz. Bölüm 6) |
| Tactical video | h264, 1920×1080, 25 fps. **199.319 kare**: mp4 başlığı, demux edilen paket sayısı ve ffmpeg tam decode (117 sn) üçü de aynı. Submission uzunluğuyla birebir uyumlu ✅ |
| Broadcast video | h264, 1920×1080, 25 fps, 6579,48 sn. **164.487 kare**: başlık, paket sayısı ve ffmpeg tam decode (152 sn) üçü de aynı. Tracking ve homografi ise **164.488** kayıt → teslim edilen videoda 1 kare eksik. Broadcast opsiyonel ve senkron değil, gönderimi etkilemez. `check_env.py` bunu `[WARN]` olarak gösterir. |
