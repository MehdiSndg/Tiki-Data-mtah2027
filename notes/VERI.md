# Veri: dosyalar, kusurlar, dış veri, bozma modeli (25 Ekim)

Eski ayrıntılar: commit `742711d` (`notes/VERI_FORMATLARI.md`, `notes/data_audit.md`,
`notes/preprocessing/degradation_model.md`, `notes/evaluation/P2_calibration.md`).

## Organizatörün dosyaları (`data/tactical/`, `data/broadcast/`)
- **`_atd.txt`**, takip: satır `kare : 46 iz (x,y,hız ; ile) : top (x,y,hız)`. Kare 1'den başlar,
  video karesi = sayı − 1. x uzun kenar (×105.3 m), y kısa kenar (×68 m), 0–1. Görülmeyen
  `NaN,NaN,NaN`. İzin kimliği yok (yuva); takımı `_atd.xml`'de. Topun üçüncü sütunu hız (karelerin
  %8.7'sinde var), yükseklik yok. Taktik 199.319 satır / 46 iz; yayın 164.488 satır / 51 iz.
- **`_atd.xml`:** takımlar (Barcelona ev sahibi, Atletic Club), 46 iz ve takımları, saha 105 × 68
  (games.py 105.3 × 68.0 kullanıyor), 1. yarıda Barça soldan sağa. Yarı sınırı alanları kullanılamaz
  (bütün video "1. yarı").
- **`_homography.csv`:** kare başına 3×3 (saha 0–1 → piksel), `reprojection_error`, `valid`. Taktikte
  %100 geçerli (hata medyan 0.24, p99 4.8); yayında %65.8 (tekrar, yakın plan, tribün). Taktik kamera
  dönüyor: homografi karelerin %98.8'inde değişiyor.
- **`_smart_tagging.xml`** (sadece taktik): 638 aralık, kod + takım + duran top türü. Possession
  değil oyun evresi (`notes/BULGULAR.md` §8).
- **Videolar:** 1920×1080, 25 kare/s, sesli (AAC 48 kHz, çok kısık). Taktik 199.319 kare
  (sha256 `ff249f07…f645f14`), yayın 164.487 kare (`3dc2a7d9…cf9f8`); yayın taktikle senkron değil
  (senkron: §BULGULAR 5).
- **`evaluation/gt/…_public.csv`:** sadece 11.185–18.684 dolu (5 dk, 17 possession), gerisi X.
- **Puanlanan bölümler:** 1. yarı 11.185–90.651, 2. yarı 110.808–190.197; dosya 199.319 satır.

## Bizim maçın kusurları (GT'siz ölçüldü)
| | ölçülen |
|---|---|
| top yok | puanlanan karelerin %40–42'si; oyun içinde ~%10–12; çeyrekten çeyreğe %27–53 (toplam), %8–12 (oyun içi) |
| boşluklar | 150 boşluk, medyan 10.7 s, p90 34 s, en uzun 221 s; kayıp karelerin %85'i 10 s'den uzun boşlukta; %88.7'si kadraj içinde |
| hareketsiz top bırakılıyor | ATD'de top yokken videoda bulunan topların %76 (ball_v1) / %88'i (hazır YOLO) < 0.5 m/s; varken %5–8 |
| ölüde hareket eden top izleniyor | public D bloklarında 20 s'ye kadar |
| top yanlış yerde | > 5 m: %1.4–4 (üç kaynak); 1–5 m: ~%6 |
| top yumuşatılmış | kareden kareye hız değişimi 0.20 m/s (DFL'in 25 kare yumuşatılmışına denk); dakikada 7.6 kırılma |
| fiziksel olarak imkânsız hız | oyun içinde 10 koşu, 140 kare; ısınmada başka toplar; 40 m/s'den hızlı adım %0.23 |
| top saha dışında | görünen karelerin %3.4–4.2'si (taç gibi gerçek durumlar dahil) |
| top kutusu boyu (ball_v1) | p10 6.8, medyan 8.2, p90 10.4 px (1920 px'lik karede) |
| saha oyuncuları | takım başına medyan 10, p5 5–6 (çeyrekte 4–9); kaybolmaların %95'i kadraj içinde; kesintisiz iz medyan 35 s |
| çift sayım | karelerin %8.6 (H) / %5.3 (A)'ünde > 10 saha oyuncusu; kopya 0.5–1.5 m yanında |
| boş slotlar | 22, 23 (H) ve 44 (A) hiç ya da neredeyse hiç yok |
| kaleciler | slot 24 (H) ve 46 (A); karelerin ~%93'ünde dolu, yarısında ekran dışında (son yerinde tutuluyor); hayalet (ekrandayken orada kimse yok): Barça %19.9, Athletic %9.6; park: slot 46 ilk 17 s ve son 214 s donmuş |
| hakemler | ATD'de yok |
| orta nokta çevresi | kickoff'ta kicker 2.0–2.2 m, top 3.3 m uzakta görünüyor |
| kamera | kadraj genişliği / saha p5 0.44, p50 0.72, p95 0.96; top kadrajın %19–77'si arasında; kamera merkezi topa medyan 23.6 m (oyuncuların ağırlık merkezi 20.3 m): topu takip etmiyor |
| nesne sayısı | puanlanan karede ortalama 20.5 (p5 14, p95 22, en çok 25) |

## Dış veri
- **DFL (Sportec Open, CC-BY 4.0):** 7 maç, ~1M kare, 25 Hz, TRACAB; her karede `BallPossession`,
  `BallStatus`, top Z'si. floodlight ile okunuyor (`mtah.external.from_dfl`). Tanımı bizimkinden
  farklı (possession ortalama 6.6–7.8 s vs GT 14.2; D %45–50 vs %19.6) → 1.6 s'den kısa ve iki yanı
  rakip olan possession'lar rakibe veriliyor (dış veriden seçildi). PathCRF deposunda ELASTIC ile
  senkronlanmış olaylar (`data/external/pathcrf/data/sportec_synced`).
- **Metrica sample data (lisansı belirsiz):** 3 maç; 1–2 CSV, 3 EPTS (bizimle aynı format, kloppy).
  Etiketler olaylardan bizim türetmemiz (`mtah.external.metrica_event_labels`): Game 1–2'de 5 dk'da
  17–18.6 possession, medyan 5.2–6.7 s, < 2 s %19–21, D %37–38. Metrica'nın takibinde ölü top = top
  yok (oyunda %3.8–6.3 eksik, ölüde %92.3–99.7); topu parça parça doğrusal. Game 3: 5 dk'da 19.8 possession, %29 ölü,
  %32 top eksik (ölçüm düzenine henüz eklenmedi).
- Diğer açık veriler (kullanılmadı): SkillCorner, PFF FC World Cup 2022, SoccerNet (NDA).

## Bozma modeli (`mtah/degrade.py`, `DEFAULT_RATES`)
Sıra: kadraj penceresi → hareketsiz top silme (≥ 50 kare, < 0.5 m/s) → kadraj içinde top kaybı
(dakikada 1.0, medyan 5 s; Metrica 0) → kaleci tutuşu (1 m, 0.5 s) → kayma (%2.5, 1–2 s) ve sapma
(%6, 1–5 m) → isteğe bağlı yumuşatma (`smooth_frames`, varsayılan 0; 25–31 kare ATD'ye yakın, tek
genişlik iki ölçüyü birden tutmuyor) → iz parçalama (medyan 35 s, %10 kayıp) → çift sayım (0.5) →
hayalet kaleci (%15, ortalama 4.5 s). Kalibrasyon: oyun içi top kaybı bozulmuş DFL %12.1 / Metrica
%10.9 / bizde %12.2; toplam top kaybı veri setinin kendi ölü top payını izlediği için ölçüt olamaz.
Uygulama kararları (P2): dış veride kaleci = 1. yarıda ortalama x'i en uçtaki slot; kadraj sadece x
boyunca bir pencere (yamuk modellenmedi); kadraj dışındaki kaleci son konumunda tutulur; Metrica
sahası 105 × 68 (bazı araçlar 106 × 68; mesafelerde ~%1); `public_gt` bayrağı ve
`control(dead="truth")` koruması public etiketin sızmasını engeller.
