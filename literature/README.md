# Literatür (21 ve 24 Ekim taramaları, tek dosya)

PDF'ler `papers/`, metinleri `text/` (ikisi de gitignore'da; `./fetch_papers.sh` yeniden kurar,
Borghesi 2023 elle indirilir, link `sources.tsv`'de). BibTeX `references.bib`. Eski konu notları
(01–09) commit `742711d`'de: `git show 742711d:literature/notes/<dosya>`.

## Possession, tracking'den
- **Vidal-Codina ve ark. 2022** (Sports Eng.; MIT + FIFA): kural tabanlı. Possession zone R_pz
  (0.5 m temiz veri, 1 m diğerleri), duel zone 1 m; kazanım için yakınlık yetmez, yön (cos 10°) ya
  da hız (5 m/s) değişmeli; Savitzky–Golay 7 kare. Set piece toleransları: kickoff 3–5 m, penaltı
  2 m, korner 3 m, taç ve aut 0.5 m. Pas P/R > %90. Sınırlaması: top verisi ve in-play bilgisi şart;
  öneri: düdük sesi, video, top z'si.
- **Link & Hoernig 2017** (PLOS ONE): oyunda her zaman bir takım possession'da, ölüde kimse; takım
  possession'ı rakibin ilk bireysel possession'ına kadar sürer (pas havadayken atanda). Algoritma:
  mesafe < T_P ve en yakın, artı top ivmesinde ≥ 4 m/s² tepe. 60 Bundesliga maçı, F = .83.
- **Borghesi ve ark. 2023** (ESWA): aynı 3 sınıflı problem (dead/home/away), iki dallı TCN +
  self-attention, 64 kare pencere (acausal), "eksik" ve "saha dışı" bayrakları. Dead/alive %89.2,
  possession %86.2 (kare doğruluğu, en temiz 8 maç, insan operatör etiketi); sadece topla
  dead/alive %83. Kıyas: Link %64.5, Morra 2020 %79.1, Khaustov 2020 %75.4, Wei 2013 %56.
- Bischofberger 2024: kural tabanlı pas/şut F 0.71 / 0.65; zorluk zamansal. Sanford 2020: video ve
  trajectory tamamlayıcı, olaylar < 0.5 s çözünürlükle.

## Top görünmezken
- **PathCRF** (Kim ve ark., KDD 2026; kod ve DFL'de eğitilmiş ağırlıklar açık): oyuncular grafın
  düğümleri + 4 dış düğüm; possession bir kenar; set-attention + maskeli CRF + Viterbi. Sadece DFL
  (5/1/1 maç), sadece in-play. Topsuz kare bazında takım possession'ı %92.4; çapraz lig F1 düşüyor.
  Bizde yeniden üretildi (`notes/BULGULAR.md` §3).
- **Ball Radar** (KDD 2023): önce topu tutan oyuncu, sonra top konumu; Metrica Game 1–3 kullanıyor;
  oyuncu possession doğruluğu %64.7.
- **TranSPORTmer** (ACCV 2024): durum sınıflandırma (pass / possession / uncontrolled /
  out-of-play) %88.9, topsuz %80.8; veri kapalı (LaLiga).

## Ölü top
- Ayrı bir "out of play" sınıflandırıcısı nadir; olay hattına gömülü. Sinyaller: tracking'in
  dead flag'i (bizde yok), topun yokluğu, saha dışı top, restart dizilimleri, oyuncu yavaşlaması,
  Smart Tagging, düdük, video sınıflandırıcı. Vidal-Codina'nın gerçek veride gördüğü hatalar:
  dead flag geç kalıyor, oyunda top eksik, taç atan oyuncu takipte yok.
- SoccerNet ball action spotting: T-DEED (2024 birincisi), dudek (2025 Team BAS); "Out" etiketi var.

## Videodan top
- **WASB** (Tarashima, BMVC 2023, MIT): HRNet ısı haritası, 3 kare girdi/çıktı, 1.5 M parametre,
  girdi 512 × 288. Futbolda F1 88.3 (TrackNetV2 86.6, MonoTrack 85.2, DeepBall 44.5); futbol
  ağırlıkları (ISSIA-CNR) açık: WASB, DeepBall(-Large), BallSeg, TrackNetV2, ResTrackNetV2,
  MonoTrack. Bizde top ~8 px, kırpma gerekiyor (plan: `notes/DURUM.md`).
- TrackNet (2019; V3 düzeltme modülü, V4 hareket dikkat haritası), FootAndBall (geniş açı, bizim
  kameraya en yakın kurgu), TOTNet 2025 (örtülme), FOOTPASS 2025 (tracking prior'lı aksiyon).
- Kendi kendine etiketleme: ATD topunu homografiyle videoya projekte edip ince ayar (ball_v1 böyle
  yapıldı; elle etiket yok).
- Genel: YOLO26, RT-DETR.

## Zamansal segmentasyon
- spellF1 = TAS'taki segmental F1@IoU. Aşırı bölünmeye karşı: MS-TCN smoothing loss (τ = 4,
  λ = 0.15), ASRF sınır dalı (+16 puana kadar), HSMM (süre öncülü), maskeli CRF + Viterbi.
- Pratik sıra: 3 durumlu HMM + geçiş cezası → HSMM (süreler DFL'den) → öğrenen model + CRF.
  Parametreler public'e göre seçilmez.

## Takım ataması, takip, kalibrasyon (SoccerNet GSR 2024–2025)
- Forma: SigLIP + UMAP + KMeans (Roboflow sports); KIST (2025 birincisi): forma rengi kümeleri +
  kümelerin ortalama x konumu. Bizimki Lab/ton + gradient boosting, %97.
- Takip ve ReID: Deep-EIoU + OSNet; kalibrasyon: RT-DETR ile saha çizgileri. Bizde ATD ve homografi
  veriliyor; öncelik düşük.

## Az etiketle öğrenme
- Snorkel: gürültülü etiketleme fonksiyonlarını GT'siz birleştiren bir label model; HMM/CRF ile
  zaman dizisine. Dış veri yaklaşımına tamamlayıcı.

## Possession'ın tanımı
- **Metrica'nın tanımı** (`events-definitions.pdf`, 2019): possession dokunuşla başlar; etkisiz
  rakip müdahalesi değiştirmez; biter: başka oyuncu alınca, şut sapınca, top hakemden sekince, top
  dışarı çıkınca. Oyun dışı: top çizgiyi geçti ya da hakem durdurdu. `Intervention: FORCED`:
  avantajdan sonra duran oyunda D düdükte başlar. Olay türleri: SET PIECE, RECOVERY, PASS,
  BALL LOST, BALL OUT, SHOT, FAULT RECEIVED, CHALLENGE, CARD.
- **FIFA EFI 2023:** dört durum (A, B, **In Contest**, Out of Play); hava topu düellosunda temas
  serbest top başlatır; possession dizileri top kontrolü olaylarıyla başlar ve biter.
- **Hipotez:** GT "In Contest"i önceki takıma yazıyor, yeni takımı kontrolde başlatıyor (0.2–1.0 s
  gecikme, 6 vaka). Metrica'nın olay operatörü ise dokunuşta işaretliyor (RECOVERY ile temas farkı
  medyan 0.00 s). Kısa (0.76 s) possession'lar da var. GT olmadan doğrulanamaz.

## Dış veri ve araçlar
- **Sportec Open DFL** (Bassek ve ark. 2025, Sci. Data, CC-BY 4.0): 7 maç, 1.002.644 kare, 25 Hz,
  `BallPossession`, `BallStatus`, Z. Riskler: tanım farkı, alan farkı (temiz, çok kameralı).
- Metrica sample data (3 maç; Game 3 EPTS, bizimle aynı format; lisans belirsiz), SkillCorner open
  data, PFF FC World Cup 2022, SoccerNet (NDA).
- Senkron: ETSY (2023), ELASTIC (2025; PathCRF'nin DFL olayları bununla).
- Araçlar: kloppy (repo 3.18.0 pinliyor), floodlight, databallpy, unravelsports.

## 25 Ekim: baştan sona tekrar okuma ve yeni tarama
- **PathCRF v2 ek B:** DFL'de LOMOCV kenar doğruluğu 66.75 ± 2.76. Bizim J03WMX yeniden eğitimimiz 0.656,
  yani aralığın içinde: eğitim sadık, "kontrolü geçemedi" sonucu duruyor. Sağlayıcılar arası düşüş büyük
  (K League → DFL olay F1 66, tersi 49).
- **Bischofberger 2024:** Metrica (ve kısmen Stats, Subsequent) verisinde top, tam olarak topa sahip oyuncunun
  üstünde: top olaylardan kurulmuş. Bu yüzden Metrica'daki dış ölçümlerimiz şişkin; dürüst vekil DFL.
- **Ball Radar ek uygulaması:** topun %5 / %10 / %20'si görülünce takım doğruluğu 91.5 / 95.0 / 97.4
  (topsuz 85.9). Kısmi topla beslenen öğrenen model boşlukları doldurabilir; ama onların maskesi rastgele,
  bizim boşluklarımız uzun ve bitişik.
- **Peral 2025 (VISAPP) ve Peral 2026 (WACV, arXiv 2512.19528), Kognia:** bizim gibi taktik kamera. Oyuncu
  kırpıntısından "top bunda mı" sınıflandırıcısı (tüp AUROC 99.8); kırpıntı + konum ile topa sahip olan
  oyuncu doğruluğu 89.6, sadece konumla 82.1. Kod ve veri yok. Bize uyarlaması: ATD topunun olduğu
  karelerden kendi etiketini üreten kırpıntı sınıflandırıcısı (elle etiket gerekmez).
- **WASB'ı eksik denedik:** makalenin çıkarımı, ısı haritası toplamını güven sayan CoH, çevrimiçi takip
  (son 3 seçimden konum tahmini, uzak adayları atma) ve oversampling'den oluşuyor. `p5l` her 10. karede,
  takipsiz çalıştı. Aynı takip filtresi ball_v1'in titremesine de uygulanabilir.
- **Etiket zamanlaması:** Link'e göre canlı insan kaydedicilerin sadece %52'si 0.6 s içinde doğru;
  possession dokunuşta değil, top oyuncunun etki alanına girince başlıyor. Vidal-Codina'ya göre olaylar
  ortalama 1.5 s geç işaretleniyor. Borghesi'nin etiketleri de canlı üç düğmeli operatör, yani GT'ye
  benzer süreç.
- **Borghesi ayrıntıları:** 1 s ileriye bakmak (hedef kare 48/64) faulleri yakalamaya yarıyor. Sadece topla
  dead/alive doğruluğu %83.6; ölü top için hız ve saha dışı bayrağı en önemli girdiler.
- **ASRF iyileştirmesi:** sınır dalının ayırdığı her bölümde çoğunluk sınıfı. Dokunuş ya da değişim
  dedektörüyle doğrudan kullanılabilir.
- **SoccerNet 2025 Team BAS ve 2026 PCBAS:** kare bazında "top hangi takımda" tahmini ya da takım
  tutarlılığı zorlayan ekipler üstte. 2026 PCBAS birincisi Macro-F1 58.94 aldı (TAAD + DST, arXiv
  2607.07320).
- **DFL BallStatus:** oyuncu değişikliğinde de "inactive".
- **Eksik oyuncu doldurma:** MIDAS (Choi 2025, arXiv 2408.10878), Jo 2025 (CIKM), Graph Imputer; FOOTPASS
  da görünmeyen oyuncuları hız ve rol öncülleriyle dolduruyor.

## Bağlantılar (künyeler `references.bib`, kaynaklar `sources.tsv`)
PathCRF arXiv 2602.12080, github.com/hyunsungkim-ds/pathcrf · WASB arXiv 2311.05237,
github.com/nttcom/WASB-SBDT · ELASTIC github.com/hyunsungkim-ds/elastic · Ball Radar arXiv 2306.08206,
github.com/pientist/ballradar · TranSPORTmer arXiv 2410.17785 · Vidal-Codina arXiv 2202.00804 ·
TrackNet arXiv 1907.03698, V4 arXiv 2409.14543 · FootAndBall arXiv 1912.05445, github.com/jac99/FootAndBall ·
TOTNet arXiv 2508.09650 · SoccerNet 2024 / 2025 arXiv 2409.10587 / 2508.19182 · FOOTPASS arXiv 2511.16183 ·
MS-TCN arXiv 1903.01945 · ASRF arXiv 2007.06866 · Snorkel arXiv 1711.10160 · Borghesi 2023
doi 10.1016/j.eswa.2023.119780 · DFL veri doi 10.6084/m9.figshare.28196177.
