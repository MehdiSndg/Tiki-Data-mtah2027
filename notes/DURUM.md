# Durum: neredeyiz, açık işler, kararlar (25 Ekim)

Bulgular: `notes/BULGULAR.md` · veri: `notes/VERI.md` · literatür: `literature/README.md` ·
teslim dokümanı: `docs/TECHNICAL.md`. Eski açık işler, gece günlükleri ve sabah raporları commit
`742711d`'de (`git show 742711d:notes/ACIK_ISLER.md`).

## Hedef ve kurallar
- More than a Hack 2027 Phase 1, BAR-ATH: 199.319 kareye H/A/D/X. Sıralama spellF1, eşit olursa
  macroF1. Geçilecek çizgi 0.2433. Son tarih **17 Kasım 2026**. Kullanıcının hedefi birincilik.
- Teslim: CSV (`<takım>.csv`), açık kod deposu (aynı dosyayı birebir üretmeli), kısa teknik doküman.
- Bizim kurallarımız: public 5 dakikaya göre seçim yok; elle etiket yok (kullanıcıya sormadan
  hiç); maç verisi, GT, kareler, önbellek, ağırlık, PDF git'e girmez; `main`'e dokunma, push yok;
  her bulgu birden fazla kaynakla; 0.01'in altındaki farklar en az 3 tohumla; araç seçmeden önce
  literatürdeki alternatifleri göster.

## Aşamalar
| aşama | ne | durum |
|---|---|---|
| EDA | veri ve GT'nin anlamı | bitti |
| P1 | ön işleme: dönemler, kickoff, top temizliği, kaleci bayrağı | bitti |
| P2 | ölçüm düzeni: dış veri, bozma modeli, harness | bitti |
| P3 | video: ball_v1, forma rengi, kaleci kurtarma | bitti (P3 kararı çelişkili, madde 16) |
| P4–P5 | anlama ve doğrulama: puan nereye gidiyor, dokunuş, PathCRF, doğrulama tablosu | bitti |
| sıradaki | modelleme: öğrenen takım kararı | plan taslağı bu gece |

Dal: `p5-env-cu128` (`p3-video-layer` üzerinden). Ortam: torch 2.7.1+cu128, `uv.lock`; 101 test.

## Şu anki teslim zinciri (v11, 26 Ekim)
`make_submission.py` (v11) → `evaluation/results/v11_chain.csv` (git'e girmiyor). **Colab'da üretildi** (ağır iş
bilgisayarda yapılmaz); dosya Drive `mtah2027/results/v10_chain.csv`, modeller Drive `mtah2027/sub_models/`.
v9 + takım kararı GRU, TCN ve oyuncu kümesi GRU'sunun ortalaması (v11; v10 GRU+TCN), ikisi de DFL + Metrica 1–3 (takım değişimli) ve kısa possession'lar
×2 ağırlıkla; sınır GRU'su da Metrica'lı. Ölü top v6 (HistGB + GRU, üç bozma çekilişi).
**Başlık: taşınan ölü toplu DFL tam zincir v6 0.666 → v9 0.699 → v10 0.717** (P7x, yeni tohumlar, 6 / 7); Metrica 1–3 tam
zincir v9 takım modeli 0.731 → v10 0.759 (P8a, 3 / 3). v11: takım kararı +0.006 (P8b, 6 / 7); **tam zincir aynı koşuda
DFL v10 0.710 → v11 0.717 (P8k, 6 / 7), Metrica v10 0.765 → v11 0.773 (3 / 3)**; public kontrolü 0.7647. Public
kontrolü 0.7625 (seçim değil). Bizim maçta 303 possession. Not: `external.py` değişince model özeti (hash) değişir;
yeniden üretmeden önce Drive'daki modeller yeni özet adına taşınmalı, yoksa yeniden eğitilir.

## Açık işler (numaralar eski listeyle aynı)
| # | iş | durum |
|---|---|---|
| 1 | Dedektörün top izini temizlemek (titreme: dakikada 111 kırılma; havadaki top) | açık; dokunuş fikrinin bizim maçta çalışmasının ön koşulu |
| 2 | Takımı videodan vermek (topa en yakın kişi kutusu + forma rengi) | **açık, artık bekletilmiyor** (25 Ekim onayı); madde 16'nın sınama yolu |
| 3 | Metrica 3. maç | bitti; ölçüm düzenine eklenmesi modelleme aşamasında |
| 4 | DFL olayları (ELASTIC) ile dürüst dokunuş testi | bitti |
| 5 | Skorlayıcının geçiş raporu | bitti (dış veride) |
| 6 | RULES.md "küçük elle etiket serbest" ↔ FAQ "No" çelişkisi | teknik dokümana; biz etiketlemiyoruz |
| 7 | 1. yarı sonu | bitti (59:07.7, POSTH) |
| 8 | ATD'nin kayan topuna dedektörle veto | bitti: etkisi yok |
| 9 | Çift sayımın etkisi | bitti: etkisi yok |
| 10 | Doğrulama tablosu | bitti → `notes/BULGULAR.md` |
| 11 | Alternatif araçlar: WASB, TrackNet; tablo modelleri; PathCRF; forma için SigLIP vb. | tablo modelleri bitti; WASB ve model deposu ölçülüyor (ilk sonuç: hazır ağırlıklar ball_v1'in gerisinde); PathCRF ikinci maçta kontrolü geçemedi |
| 12 | Literatür ve araç taraması | bitti → `literature/README.md` |
| 13 | Katı fp32 ve ortam yükseltmesi; Blackwell'de maç boyu tarama | bitti |
| 14 | Ölü topa ek sinyaller (yayın ayrılması, hareketsiz top, yarı sonu); oyun içi boşluklarda yanlış "ölü" (%61–74) | incelendi (25 Ekim gecesi): hatalar uzun boşluklarda; boşluk özellikleri yardım etmiyor; yayın ayrılması maç boyu GT'siz doğrulandı; hareketsiz top zayıf. Modelleme |
| 15 | Video okuma hızı | bitti: darboğaz değil |
| 16 | P3 kararını yeniden aç (beş test bir hakeme karşı) | kısmen: "boşlukta ATD takımları yanlış" hipotezi çürüdü (forma %93–96 uyum); forma possession için bağımsız hakem değil. Dış veri doldurmaktan yana, Smart Tagging karşı; teslim zinciri kullanıcının kararı |

Modelleme aşamasının diğer adayları: öğrenen takım kararı (dokunuş + PathCRF, TCN), havadaki
topun alıcıya doğru kaydırılması (DFL Z ile sınanabilir), kaleci tutuşu özelliği.

## Bu gece (25 Ekim, kullanıcı hepsini onayladı; "mümkünse tüm araştırmalar")
Yeni Colab defteri, `scripts/colab_bootstrap.sh`, sırayla:
1. **WASB ve model deposundaki diğer 6 futbol modeli.** Bizim top ~8 px; model 512 × 288 alıyor.
   Seçenekler: A bütün kare (top ~2 px), B 2 × 2 karo (~4.4 px), C 4 × 4 doğal çözünürlük (~8 px),
   D ince ayar. Önce A/B/C küçük blokta; işe yarayanı doğrulama bloklarında (her 5. 10 dk'lık blok,
   public hariç, 3000 kare) ball_v1 ile aynı ölçütlerle: recall/precision (ATD'ye göre), boşluk
   doldurma, nokta yanlışı, dakikada kırılma, hız. D yok. Depo `data/external/wasb` (MIT, 923462c).
2. **Madde 16 + 2:** gerçek boşluklarda forma renginden takım; Smart Tagging'den bağımsız.
3. **Madde 14:** oyun içi boşluklarda yanlış "ölü"; dış veride inceleme ve ek sinyal denemeleri.
4. **Modelleme planı taslağı** (kod yok, sabah birlikte).

## Modelleme planı (taslak, 25 Ekim gecesi; sabah birlikte okunacak, kod yok)
**Hedef:** puanın kaybolduğu yer el değişimleri (`BULGULAR.md` §2). Kabul ölçütü her adım için
aynı: bozulmuş dış veride tam zincir spellF1, iki kaynakta da, 3 tohumda ve aralıklar örtüşmeden
en az +0.01. Public'e bakılmaz; bizim maçta sadece GT'siz akıl sağlığı kontrolü (90 dk'da ~300–330
possession, ölü top ~%35–40).

1. **Bozma modelini dürüst yapmak (önce).** Topu 25–31 kare yumuşat (ATD gibi), dedektör topu için
   titreme ve havadaki top kayması ekle, top ile topun yanındaki oyuncuları çekişmeli anlarda
   birlikte kaybet. Yoksa dış verideki kazanç bizim maçta olduğundan büyük görünür (P4g dersi).
2. **Top kaynağı.** WASB sonucu ve madde 16'ya göre: ATD + ball_v1, takım kararı için sadece
   güvenilir kareler (madde 16'nın cevabına bağlı). Madde 1: dedektör izini temizlemek (dokunuşlar
   arasında düz yol varsayan sağlam bir iz modeli; havadaki topu ayıklamak), hedef dakikada
   30–47 kırılma.
3. **Takım kararı modeli.** Alternatifler, ucuzdan pahalıya:
   | aday | artısı | eksisi |
   |---|---|---|
   | kontrol + dokunuş kuralı | dış veride şimdiki en iyi (+0.10–0.15), açıklanabilir | yumuşak topta çöküyor |
   | kare sınıflandırıcı (GBM) + HMM/HSMM çözümleme | ucuz, geçiş cezası possession sayısını doğrudan ayarlar | özellik mühendisliği |
   | TCN (Borghesi 2023) | aynı problem, iki dallı | 7+2 maç az olabilir |
   | PathCRF (topsuz) | top kayıpken çalışıyor; ağırlıklar hazır | kimliksiz izlerle uyumsuz; 2 test maçında kontrolü geçemedi |
   | MS-TCN / ASRF sınır dalı | spellF1'in kendisini hedefler | en pahalı |
   Önerim: GBM + HMM ile başla (özellikler: iki takımın topa en yakın mesafeleri ve hızları,
   kırılmalar, topun hızı ve yön değişimi, topun var olup olmadığı), PathCRF'yi topsuz karelerde
   bir özellik olarak ekle. DFL 5/1/1 bölme + Metrica testi.
4. **Ölü top.** P5n'in boşluk özellikleri, yayının sahadan ayrılması (sadece bizim maç), kaleci
   tutuşu (E29), hareketsiz top.
5. **Teslim zinciri ve tekrar üretim.** Tek komut, katı fp32, dedektör ağırlığının nasıl
   sağlanacağı (yayımlamak ya da eğitimi yeniden üretilebilir kılmak), teknik doküman.
Takvim (17 Kasım'a ~3 hafta): 1. hafta adım 1–2, 2. hafta adım 3, 3. hafta adım 4–5 ve doküman.

## 25 Ekim gecesinin sonuçları (ayrıntı `BULGULAR.md`)
- **WASB ve model deposu (hazır ağırlıklar):** ball_v1'in gerisinde. En iyisi WASB 2 × 2 karoda %49 /
  %69 / %40 (ball_v1 %68 / %80 / %52); bütün karede %0; doğal çözünürlükte isabet %15; titremeyi de
  azaltmıyor. İnce ayar (D) denenmedi.
- **P3 kararı:** "boşlukta ATD takımları yanlış" çürüdü (forma %93–96). Forma rengi possession için
  bağımsız hakem değil (ilk yorumum yanlıştı, düzeltildi). Doldurmanın lehine dış veri, aleyhine
  sadece Smart Tagging. **Smart Tagging yapışkanlığı ödüllendiriyor** (T taraması: zirvesi 0.8 s, dış
  verinin 0.4 s). Kaleci düzeltmesinin yararına bağımsız kanıt yok.
- **Ölü top:** hatalar uzun boşluklarda; boşluk özellikleri yardım etmiyor; bizim maçta yayın
  sinyali uzun boşluklardaki "ölü"lerin %67'sini doğruluyor, %4'ü muhtemelen yanlış.
- **Diğer sınamalar:** R × T çözüldü (yumuşak topta 1.5/0.4); hareketsiz top zayıf; yayın ayrılması
  maç boyu güçlü; kaleci tutuşunda sınıflandırıcı bizim maçta iyi; bozma modeli etiketleyici
  çıktısında gerçekçi; dedektör titremesi ikinci ölçümle doğrulandı; aynı kartta tarama maç
  ölçeğinde bit bit tekrar üretiliyor.
- **PathCRF ikinci maç (J03WMX) bitti:** kontrolü geçemedi (ham 0.530 vs 0.534, birleştirilmiş
  0.575 vs 0.613); iki maçta tutarlı üstünlük yok.
- **Senden beklenen kararlar:** (1) dedektör/yayın topunu takım kararına almak; (2) WASB'a ince
  ayar (D) yapmak mı; (3) modelleme planı taslağı.

## Kaldığım yer (27 Ekim sabahı)
- Teslim v9 (`make_submission.py`), taşınan ölü toplu DFL tam zincir 0.699; dosya ve ağırlıklar yerelde.
- Gece kazandıranlar: takım değişimi (P7e/P7f/P7n), ölü top HistGB+GRU ortalaması (P7c), ölü top için üç bozma
  çekilişi (P7h). Kazandırmayanlar: SkillCorner, top doldurma, oyuncu kümesi, y aynası, ikinci çekiliş, 5 tohum,
  büyük GRU, Metrica, ceza ve bloklama ayarları, kendi kendine öğrenme, beş çekiliş.
- Kalan açık kısa possession'lar (P7r): 0–2 s'nin %28'i, 2–5 s'nin %58'i bulunuyor. P7s (kısa possession'lara kayıpta ağırlık, +0.004, 4 / 7) ve
  P7t (dedektör kanalı, −0.002) gürültü içinde.
- Colab: arka planda kuyruk koşucusu (`pkg/runner.py`), işler Drive `mtah2027/queue/*.sh` ile veriliyor.

## Hatırlatma (26 Ekim)
- Kullanıcı "video ile possession" hakkında bir şey söyleyecek: işler bitince sor.

## Kullanıcının verdiği kararlar
- 22 Ekim: kickoff top toleransı 4 m; kaleci için "doğrulanmadı" bayrağı + YOLO kişisinden kurtarma.
- 22 Ekim: elle etiket kuralı katı kalıyor (değerli tek seçenek ileride: public dışında 20–30 dk'lık
  ölü top doğrulama seti; sadece seçim için, konuşulacak).
- 23 Ekim: madde 1 ve 2 bekletilsin → 25 Ekim: madde 2 açıldı.
- 23 Ekim: organizatöre soruları kullanıcı soracak (Metrica lisansı, ekran kartı farkı).
- 24 Ekim: ortam torch 2.7.1+cu128'e yükseltildi.
- 25 Ekim: her şey Colab'da yapılabilir; notlar az ve kısa (61 md dosyası 5'e indirildi).

## Colab işletimi
- Bağlantı: `colab-mcp` (`~/.local/bin/colab-open`; adres `/tmp/colab-mcp-url.txt`). Bağlandıktan
  sonra `open_colab_browser_connection` iki kez; kullanıcı çalışma zamanını kendisi seçer, sonra
  ekran kartı kontrol edilir.
- Hazırlık: Python hücresinde `drive.mount('/content/drive')`, sonra
  `!bash /content/drive/MyDrive/mtah2027/pkg/colab_bootstrap.sh` (repo, ortam, veri + sha256,
  önbellekler, Blackwell taraması; son satır BOOTSTRAP_DONE).
- Uzun işler `nohup` ile, çıktı ve kayıt Drive'a (`mtah2027/runs`, `mtah2027/logs`); bekleme
  süreç numarasıyla ya da bitiş satırıyla (`pgrep -f` kendini bulur).
- rclone (`gdrive:`, drive.file izni) Colab'ın yazdığı dosyaları göremiyor; sonuçlar hücre
  çıktısıyla (gerekirse tar + base64 + sha256) bilgisayara alınıyor.
- Drive: `mtah2027/pkg` (bundle, ağırlıklar, bootstrap), `data`, `cache`, `runs`, `logs`, `backup`.

## Kullanıcıyla anlatma turunun kalan konuları
Kaleciler, kamera, Smart Tagging kusurları, GT ayrıntıları, dış veri, bozma modeli, ölçüm düzeni,
P1–P4, şu anki çözüm, kurallar (kullanıcı "anlat" deyince, teker teker).

## Hatırlatmalar
- Takım: 5 kişi (kural 3–5 tamam); bir süre herkes ayrı çalışıyor, en iyi skorun üstünde sonra birlikte (26 Ekim).
- rclone'un ortak Google client_id'si 2026 içinde kapanıyor.
- Disk: uv önbelleği ~28 GB; silmek kullanıcının onayına bağlı; WSL sanal diski Windows'tan
  sıkıştırılmalı.
- Kullanıcı sudo şifresini değiştirmeli.
- Teknik dokümana: madde 6, Ultralytics AGPL-3.0, dedektör ağırlıklarının nasıl elde edileceği
  (yayımlamak ya da eğitimi yeniden üretilebilir kılmak; açık soru).

## Devam noktası (27 Eki 06:20)
Teslim v11 (DFL tam zincir 0.717, Metrica 0.773). Colab'da arka planda (loglar Drive `mtah2027/logs/`): P8n DFL
(çok görevli, 5 maçta 4 / 5 önde) ve Metrica hakemi `p8n/metrica.log`; P8j Metrica hakemi `p8j/metrica.log` (v12 = ST
ön eğitimi, DFL +0.009 5 / 7; v12_chain.csv üretildi, GT'siz başa baş); P8p (çok görevli + ST) `p8p/dfl.log`,
`p8p/metrica.log`. Karar: bir aday DFL'de ve Metrica'da ikisinde de önde ise teslime al (v12 ya da çok görevli için
make_submission sürümü yazılacak); değilse v11 kalır.

## Plan (27 Eki, araştırma raporu `reports/Possession modeli iyileştirme fırsatları.md` sonrası)
Karar kuralı: aynı koşuda eşli karşılaştırma, DFL ve Metrica ikisinde de önde (iç içe seçim); GT'siz hakemler possession
ekleyen değişikliklerde veto değil (o yönde yanlılar); public hiçbir karara girmez. Koşudan koşuya fark ±0.007–0.01.
- #0 Model özeti setgru / deadball / m1 / possession_metrics'i, ölü GRU CARRY_RATES'i, frames games.py'yi kapsıyor. **Bitti.**
  Modeller artık elle yeni özete taşınmıyor; özet değişince Colab'da yeniden eğitilir.
- #1 OOF dökümü (`analysis/p9a_oof.py`, Drive `mtah2027/oof/oof_<maç>.npz`, 5 paralel iş). **Sürüyor.**
- CPU, OOF'tan: #2 ceza topluluğa göre (sınırlı 4/2, 6/2, eps, −log b, üye ağırlıkları), #7 sınır tepelerinde segment
  oylaması, #3 ölü top kaybını türlere ayırmak + birleşim sonrası boşluk kapatma, histerezis, aynı takım vetosu.
- GPU, paralel: #4 P8n/P8j/P8p birleşimi, #5 dead_carry_p 0.25/0.5/0.75 + gerçekçi taşıma, #8 sınır kaybı (R 5/15,
  pozitif ağırlık, T-MSE), #9 eğitim hijyeni (rastgele pencere başlangıcı, yarı sonu, TCN genişliği), #10 kaleci
  özellikleri, #12 ±20–80 s ölü top bağlamı.
- #6 ELASTIC etiket denetimi (3 DFL maçı, pas mı alış mı). #11 teslim zinciri = ölçülen zincir (ölü top Metrica 3,
  yayın kuralının GT'siz ablasyonu).

## Teslim v14 (27 Eki akşam)
`evaluation/results/v14_chain.csv` (Drive `mtah2027/results/v14_chain_cdb4ac8.csv`, modeller `sub_models/`, özet
b0ff8105…). Başlık: **DFL tam zincir 0.725** (aynı koşuda v11 0.717), Metrica 1–2 0.799. Sıradaki: Metrica 3 teyidi,
P9e karışık bozma (ilk maçta 6 / 6 koşulda önde), ölü top kenar yerleşimi (kaybın yarısı), P9h birleştirme eşiği.

## Takip listesi (27 Eki 19:45; arkadaşın bulguları + araştırma raporu)
Arkadaş: karışık bozma → P9e sürüyor (ilk maç 6/6 önde); β = ceza ölçeği → P9b'de kapsandı; ince hedef → P9g bitti,
P9h (birleştirme 0 / 0.8 / 1.6) bekliyor; dokunuş dizisi → yapılmadı (düşük); seed seti entropi eşleme → OOF üzerinde
kalibrasyon taraması yapılacak; ortak protokol → OOF'lar hazır, arkadaştan aynı biçim istenecek.
Rapor: #0 ✓, #1 9/10, #2 ✓ (v14), #3 ✓ ayrıştırma + P9i kenar düzeltici sürüyor, #4 v14 ilk adım, #5 ✓, #6 ✓,
#7 ✓ (Metrica +0.006, DFL nötr), #8 / #9 / #10 GPU sırasında, #11 kısmen (yayın kuralı ablasyonu, ölü topa Metrica 3),
#12 ✓ (Z), #13 sonra. Açık çelişkiler: bizim maçta az bölme; possession sayıları DURUM / BULGULAR.

## Gece planı (27 → 28 Eki), sırayla; GPU'da aynı anda 5–6 iş
1. Biten işlerin kararı: OOF Metrica 3 → v14'ü 10 maçla teyit (P9b); P9e olumluysa v15 (karışık bozmayla eğitilmiş
   üyeler) üret ve teslime al; P9k (resmi baseline'larla hakem ayarı) → birleştirme eşiği, gerekirse P9h ile doğrula.
2. P9j: top dolgusu simülasyonu GRU ile (kayıp topun %30 / %60 / %100'ü geri gelirse); ikinci arkadaşın dedektörü
   (%40.7 → %27.3) ve bizim ball_v1 dolgusu için karar.
3. Özellik paketi: kaleci özellikleri + dokunuş epizotu özellikleri (iki arkadaş da önerdi).
4. Eğitim hijyeni (#9) + sınır kaybı varyantları (#8).
5. CPU: sıcaklık / entropi kalibrasyonu (OOF), yayın kuralının GT'siz ablasyonu, P8p devam, dokunuş dizisi.
İkinci arkadaş (takip listesine): resmi baseline'larla hakem ayarı → P9k sürüyor; videodan maça özel top dedektörü
(%40.7 → %27.3) → P9j simülasyonu, olumluysa dedektör çıktısı istenecek / ball_v1 dolgusu; dokunuş epizotu özellikleri
(+0.012) → özellik paketi (madde 3); ATD benzeri oyuncu gürültüsü → bizde var (parçalı iz, kopya, hayalet kaleci;
oranları onların %7.3 kayıp payıyla karşılaştırılacak); 2. yarı başlama öncesi D → bizde var.

## Devam noktası (28 Eki öğle; Colab işlem birimi bitti, VM kapandı)
Teslim v14 (DFL tam zincir 0.725, `evaluation/results/v14_chain.csv`). Güvende: OOF 10 maç (Drive `mtah2027/oof/`),
log'lar, BULGULAR. Kaybolan: yarım kalan v15 üretimi (karışık bozma; 3 / 6 ağ eğitilmişti), P9n Metrica.
Gece sonuçları BULGULAR'da: P9e karışık bozma (normal nötr, sert +), P9k hakem ayarı (GT ≈ 3 s, taşıma 0.4), P9m kaba
hedef alınmadı, P9j top dolgusu (kusursuz +0.021, kısa boşluk +0.004), P9n ≈ 0, P9o / P9p / P9q / P9r alınmadı, P8p ≈ 0.
Sıradaki: GPU seçeneği (Kaggle ücretsiz / Colab ek birim / saatlik kiralık / yerel 4060 — kullanıcı kararı); sonra v15
üretimi + MTAH_DUMP ile bizim maçta ceza ↔ possession sayısı taraması (az bölme: v14 254 vs ayarlı ~286–300 / 90 dk).

## Kaggle (28 Eki): Colab işlem birimi bitti, işler Kaggle'da
Hesap mehdinda (telefon doğrulandı: 2× T4 + internet). Özel datasetler: mtah-code (bundle; `scripts/kaggle/push.py --code`),
mtah-ext (DFL, Metrica), mtah-match (BAR-ATH videosuz + teslim önbellekleri), mtah-video (iki video), mtah-extra (Drive
cache, tüm modeller, OOF, ST / SC önbellekleri, dedektör ağırlıkları), mtah-local (PC'ye özel önbellekler, PathCRF).
İş gönderme: `uv run python scripts/kaggle/push.py <slug> '<ad>=<komut>' ...` (her iş bir GPU'da; kurulum ~11 dk; çıktı
`kaggle kernels output mehdinda/<slug> -p <dir>`). Kurulum testi 141 test geçti. Haftalık ~30 GPU saati.
Çalışan: mtah-v15 (v15 ve v14 aynı koddan, MTAH_DUMP ile bizim maçın olasılıkları).
Sonuç (28 Eki akşam): v15 üretildi, GT'siz hakemler çelişkili (ST v15, forma v14), normal hakem nötr → teslim v14 kalıyor,
v15 yedek. Bizim maçta az bölme kod çözmeden değil (P9s), girdi / alan kaymasından: sıradaki araştırma oradan.


## Plan (29 Eki; ikinci araştırma `reports/Yapısal sıçrama adayları.md` sonrası)
Hedef: 0.80'i geçmek ve birincilik ("yaklaşmak" değil, 7 Kas); skorda tavan yok. Ölçülmüş kaldıraçlar: ölü top +0.025, kusursuz top +0.021; adaylar bunların üstüne eklenerek sınanıyor.
Kaggle aynı anda en çok 2 GPU oturumu çalıştırıyor. Sıra:
1. P9u (oyuncu sayısı kayması; `mtah-p9u`) ve P9w (dokunuş penceresi 3 → 12; `mtah-p9w`): çalışıyor.
2. P9y (uzun boşluk dolgusu: oyundaki mi ölü mü, gürültü 1–8 m): kod hazır, ilk boşalan oturuma.
3. P9x (oyuncu kümesine possessor başı, `mtah/setwho.py`): kod hazır, sonraki oturuma.
4. Bozma eşlemesinin 2. turu (P9u sonucuna göre), sonra kazananlarla v16.
Elenen: P9v (restart kuralı, kenar oturtma), P9z (kenar kazancı düdüklü duruşlarda), video ölçeği, TTA, PathCRF yeniden eğitimi.
Kabul kuralı: DFL ve Metrica birlikte, ayrıca 3 s hakeminde de geriye gitmemeli.

## Gece sonu (29 Eki sabah)
Bu gece hepsi ölçüldü, hiçbiri alınmadı: P9u (oyuncu sayısı), P9v (restart kuralı / kenar oturtma), P9w (dokunuş penceresi),
P9x (possessor başı), P9z (kenar kazancı düdüklü duruşlarda), P9aa/P9ab (oyunculardan top tahmini ~10 m), P9ac (videonun
uzun boşluk payı küçük). Tek açık ipucu P9y: uzun boşlukların oyun karelerinde gerçek top değerli, ama bizim maçta bu ~1.4 dk.
Çalışan: `mtah-p9ad` (3 → 6 tohum). Sonucu olumluysa v16 = v14 + daha çok tohum (kod riski yok).
Teslim hâlâ v14.
P9ad alındı → v16 (v14'ün altı takım ağı 6 tohumla, `make_submission.py v16`). Kaggle `mtah-v16` kuruyor (a: GRU/TCN üyeleri, sonra
teslim; b: set üyeleri, sınır GRU, ölü top). Gelince: v14k ile fark, GT'siz yargıçlar, possession sayısı; sonra teslim v16.
**Teslim v16** (29 Eki): `evaluation/results/v16_chain.csv`. `mtah-v16b` ikinci kurulum **bayt bayt aynı** dosyayı verdi (Kaggle T4'te
belirlenimci). Ağırlıklar yerelde `models/` (`*955b45483d366b83*`, ölü top `dead_gru_d012_c697fd6a9a61caea.pt`).

## Devam noktası (30 Eki)
Teslim **v16** (bayt bayt yeniden üretilebilir; ağırlıklar `models/*955b45483d366b83*`, döküm `cache/v16_dump.npz`).
Son sonuçlar BULGULAR'da: P9ae / P9ag / P9ah (tohum başka yerde işe yaramıyor), P9ai (ölü top bağlamı −0.008), Smart Tagging
geçişleri (zaten yakalanıyor), P9aj (yayın kuralı kalıyor).
Çalışan: P9ak yerelde (`scripts/run_local.sh`, log `eda_out/logs/p9ak_local.log`; 3 olağan + 3 karışık bozmalı ağ vs 6 olağan;
J03WMX +0.0005, 3 s'de +0.014; J03WN1 / J03WOH sürüyor). Kalan 7 maç (J03WOY, J03WPY, J03WQQ, J03WR9, metrica) Kaggle kotası
yenilenince `push.py mtah-p9ak` ile (iki iş). Kazanırsa v17 = v16'nın her üyesi 3 olağan + 3 karışık tohum.
Yerel işler daima `scripts/run_local.sh` ile (12 GB sınır); WSL yeniden başlarsa /tmp silinir, sonuçları cache/ ya da eda_out/'a yaz.

## Karar (3 Kas)
SoccerTrack v2 video dokunuş modeli ve yayın kırpıntıları kapandı (kullanıcı: 1–2 hafta, içinden çıkılmaz; kırpıntı
kanıtının etkisi zaten sınırlı). Çalışan: Kaggle `mtah-p9ak` / `mtah-p9ak2` (P9ak kalan 9 maç), Colab f8a86b6 altı iş
(P9al v16 OOF → Drive `mtah2027/oof_v16/`, P9am büyük GRU; loglar `mtah2027/runs/f8a86b6/`). Sonra: v16 OOF ile CPU
taramaları, arkadaşlarla birleşim (ortak OOF biçimi).

## Gece (3 → 4 Kas)
Colab havuzu (`scripts/colab_pool.py`, Drive `mtah2027/queue2`, en çok 6 iş): P9al (v16 OOF → `mtah2027/oof_v16`), P9am (BigGRU),
P9ao DFL (geniş TCN), sırada P9as (merkezi hız) ve P9at (BigGRU OOF üyeleri → `mtah2027/oof_big`, 5 iş). Kaggle: P9aq (yarı sonu),
P9ao Metrica. v17'ye kesin: dead_v8 (Metrica 1–3 + X maskesi), restart_evidence strict. Adaylar tam zincirde ölçülecek:
P9at bitince v16 OOF'ta GRU / çok görevli GRU yerine BigGRU (CPU), 1.6 s ve 3 s hakemleri.

## Takım deposu (4 Kas)
Private: https://github.com/MehdiSndg/mtah2027 (`team` remote; `p5-env-cu128` → `main`). Commit mesajlarındaki Claude satırları
silindi (filter-branch; dosyalar aynı; eski geçmişin yedeği `cache/git_backup/before_trailer_rewrite_20261104.bundle`). Bu yüzden
4 Kas öncesi notlardaki commit kısaltmaları artık eşleşmiyor; model özetleri (`_hash`) dosya içeriğinden, etkilenmedi.
Colab birimi bitti (4 Kas); OOF'lar yerelde `cache/oof_v16`, `cache/oof_big`. P9au yerel GPU'da (`eda_out/logs/p9au_local.log`).

## Gece (5 Kas)
Teslim **v17** (`evaluation/results/v17_chain.csv`; DFL 1.6 s ≈ 0.728 / 3 s ≈ 0.702, Metrica 3 s ≈ 0.774; public 0.763).
Yerel GPU kuyruğu (sırayla, 12 GB sınır): P9au DFL (geniş TCN OOF, `eda_out/logs/p9au_local.log`) → P9au Metrica
(`p9au_met_local.log`) → P9av büyük set (`p9av_local.log`). Kaggle: P9aw (BigGRU 3 bozma kopyası) çalışıyor; haftalık kota doldu.
Takım: GitHub release `artifacts-v16` (v16 ağırlıkları, oof_v16, oof_big). Sıradaki: P9au bitince BigGRU + geniş TCN tam zincir; v18.
Güncelleme (5 Kas 02:20): P9aw bitti (BigGRU 3 çekiliş, takım kararı DFL +0.016). Yerel kuyruk: P9au J03WPY'den sonra durur →
P9ay DFL (`eda_out/logs/p9ay_dfl_local.log`, ~1 sa / maç) → P9ay Metrica → P9av. Kod paketleri (Drive, Kaggle) HEAD'de.

## Devam noktası (7 Kas 15:40)
Teslim **v17** (DFL 1.6 s ≈ 0.728 / 3 s ≈ 0.702, Metrica 3 s ≈ 0.774, public 0.763). Aday v18: tek başlı BigGRU 3 bozma
çekilişiyle (P9ay: DFL tam zincir +0.0058 7 / 7, 3 s +0.0043; Metrica 1–2 −0.0025 / −0.0030, Metrica 3 sürüyor). Kapı
`analysis/p9ay_chain.py` (tek kural; önce yerine koyma = v18, tutmazsa 7. üye = v18b), kuyrukta kendiliğinden kurulum.
Yerel kuyruk `eda_out/queue.txt` (`scripts/local_queue.sh`, `scripts/mem_guard.sh`, `eda_out/logs/memwatch.log`): P9ay
Metrica 3 → v18 kapısı + kurulum → P9au kalan 2 DFL + Metrica → P9az TCN → P9av → P9az BigGRU 3 / 5.
Altyapı: 5–7 Kas WSL'i Windows bellek baskısıyla 4 kez kapandı → .wslconfig memory 22 → 18 GB, işler ≤ 8 GB, kuyruk
Windows'ta ≥ 4.5 GB boş olmadan iş başlatmıyor; WSL diski D:\WSL\Ubuntu'ya taşındı (C: dolmuştu). Yedek
D:\mtah_backup_2026-11-07. Araştırma: `reports/Geçmiş denetimi ve sıçrama yolları.md`.
Açık: P9aq (yarı sonu pencereleri) tam zincir testi yok; takım arkadaşlarıyla OOF birleşimi; Kaggle kotası yenilenince
paralel işler; Colab birimi yok.
- 7 Kas 18:30: P9az BigGRU 3 / 5 çekiliş yerelde sığmadı (~13 GB; swap'te kilitlendi, GPU %0) → durduruldu, Kaggle'a
  (30 GB RAM) bir oturum boşalınca: `MTAH_ARCH=big MTAH_DRAWS=3,5 python analysis/p9az_draws_member.py only=...`.

## Devam noktası (8 Kas 17:10)
Teslim **v19** (`evaluation/results/v19_chain.csv`; DFL 0.7347 / 0.7044, Metrica 0.7754 / 0.7753, public 0.718). Kahin ayrıştırması
(v19 OOF): kusursuz ölü top +0.025 (DFL) / +0.031 (Metrica), **kusursuz takım kararı +0.22 / +0.17**; 0–2 s GT spell'lerinin %34'ü,
2–5 s'nin %66'sı bulunuyor. Aynı bilgiyle üye büyütmek doydu (her deneme +0.002–0.006); kısa possession için yeni bilgi lazım.
**Sıradaki büyük iş (Colab birimi gelince):** SoccerTrack v2 videolarıyla (Drive `mtah2027/soccertrack`, 145 GB, oyun durumu karesi
= video karesi, takımlar etiketli, kalibrasyon `raw/<id>/`: homografi + 3840×1906 bozulma haritaları) possessor-kırpıntı kanıtını
gerçek GT'ye karşı tam zincirde ölçmek ve SCALE'i (bugün GT'siz 0.2) orada ayarlamak. İlk adım: birkaç karede izdüşüm yönü / bozulma
seçeneğini YOLO kişi kutularına isabetle otomatik seçmek; kişi kutuları yolo11m, kırpıntı p6d (128 px, 1.4 kenar payı),
models/possessor_r18.pt. 4 dakikalık MOT klipleri işe yaramaz (4096×1080, takım ve maç zamanı yok).
Yerel kuyruk: P9bh (iki başlı geniş TCN üç çekiliş, 4 / 7 DFL kayıtlı, kalan 3 + Metrica) → P9bj (eğitim tarifi). Kaggle kotası
~14 Kas'a kadar dolu.
- 8 Kas 22:45, SoccerTrack video planına ek (Colab gelince): GSR JSON'larında her oyuncu ve her kare için `bbox_image` (x, y, w, h;
  ~28 x 33 px), takım (left/right) ve `player_id` var → izdüşüm de YOLO da gerekmiyor. `bas/<id>/<id>_12_class_events.json`
  "actions"ında `frame`, `team`, `player_id` (117092: 2085 olay; PASS 898, DRIVE 787, HIGH PASS 124, OUT 75 ...) → olay karesinde
  dokunan oyuncunun kırpıntısı pozitif, diğerleri negatif: gerçek GT'li, dış veride possessor eğitimi (kurala uygun). Kullanıcı:
  ikinci bir emre kadar Colab dışı işler.
