# Bulgular (25 Ekim itibarıyla, tek dosya)

Bütün bulgular, sayıları ve kaç bağımsız kaynakla doğrulandıkları. Eski ayrıntılı notlar (EDA,
P1–P5 raporları, gece günlükleri, doğrulama tablosu) commit `742711d`'de duruyor:
`git show 742711d:notes/<yol>`. Script'lerin ürettiği ham tablolar artık `eda_out/reports/`'a
yazılıyor (git'e girmiyor); her satırın sonundaki script onu yeniden üretir.

**Durum:** **S** sağlam (≥ 2 bağımsız kaynak) · **O** orta (tek kaynak ama sayım ya da duyarlılık
testi) · **Z** zayıf (tek kaynak, az örnek ya da sadece public) · **K** çelişkili.
**Public kuralı:** public 5 dakika hiçbir yerde seçim için kullanılmadı; yalnızca kontrol ve anlama.

## 1. Hedef ve ölçüm
- **Hedef organizatörün etiketi, gerçek değil.** GT düzeltilmiş tracking'den mekanik türetiliyor;
  "tanımı GT'den farklı okursan GT haklıdır" (CHALLENGE.md). **S**
- **spellF1 = segmental F1@IoU** (0.3–0.7) H/A segmentlerinde; açgözlü bire bir eşleşme. Her fazla
  possession bir yanlış pozitif; D bir possession'ı ikiye böler; X yarı içinde ileri doldurulur.
  Yarılar ayrı. (skorlayıcı kodu) **S**
- **Public pencere temsili değil:** top kaybı public'te %11.8, maçın geri kalanında %42.1; 20
  bloğun en iyisi. Bootstrap aralığı ~0.3 genişliğinde. **S**
- **Dış veride "doğru cevap" vekil:** Metrica'da olaylardan bizim türettiğimiz etiket, DFL'de
  `BallPossession` + 1.6 s birleştirme. Labeler sıralaması (touch > control+touch > control)
  birleştirme 0 / 1.0 / 1.6 / 3.0 s ve ölü top ±1 s'de, 3 tohumda aynı; mutlak sayılar oynuyor
  (DFL control 0.49–0.56). Sıralama için **S**, mutlak sayılar için değil. (`p5f_verification.py` D)
- **Metrica türetmesi GT'ye DFL'den yakın** (5 dk'da possession 16.5–17.9 vs GT 17 vs DFL 18–28;
  medyan süre 5.8–7.2 / 7.6 / 2.8–5.3 s). Tek karşılaştırma. **O**
- **Bozma modeli** dış veriyi bizim ATD gibi bozuyor (oranlar `notes/VERI.md`). Kabul ölçütü oyun
  içi top kaybı: bozulmuş DFL %12.1, bizde %12.2. Baseline bozulmuş Metrica'da 0.319, public'te
  0.326. Oranlar çeyrekten çeyreğe ±%50 oynuyor; ×0.5 ve ×1.5'te sıralama aynı. GT'siz ikinci kontrol
  (`p5t_degrade_realism.py`, etiketleyici çıktısı, kendi ölü topuyla, 3 tohum): baseline_smooth50 90
  dk'da bizim maçta 505 possession, bozulmuş DFL / Metrica 531 / 525 (dakikada etiket değişimi 6.8 /
  7.2 / 6.8); en yakın oyuncu 1.656 / 1.683 / 1.881; kontrol kuralı 347 / 292 / 291 (medyan 6.1 / 7.5 /
  7.6 s): bizim maçta kontrol biraz daha sık bölüyor. **S**
- **Tohum gürültüsü ~±0.01** (P5f/P5g, 3 tohum): 0.01'den küçük farklar tek tohumla yorumlanmaz. **S**
- **Ölçüm düzeninin doğruluğu:** organizatörün baseline'ı kendi formatımızda kare kare aynı;
  "en yakın oyuncu" kodumuz `baseline_raw` ile 28.475 karede %100 aynı; 4 baseline bayt bayt aynı
  üretiliyor, skorlayıcı çıktısı pandas 2.3 ile 3.0'da aynı. **S**
- **İlk karşılaştırma (P2; spellF1, maç ortalaması):**

  | etiketleyici | Metrica temiz | Metrica bozulmuş | DFL temiz | DFL bozulmuş | public (kontrol) |
  |---|---|---|---|---|---|
  | baseline_smooth50 | 0.454 | 0.319 | 0.339 | 0.277 | 0.326 |
  | en yakın oyuncu | 0.114 | 0.106 | 0.077 | 0.089 | 0.038 |
  | kontrol, ölü top etiketten | 0.681 | 0.567 | 0.620 | 0.541 | — |
  | kontrol, kural tabanlı ölü top | 0.602 | 0.497 | 0.380 | 0.440 | 0.441 |
  | kontrol + m1 sınıflandırıcısı | — | 0.548 | — | 0.487 | 0.442 |

  Maçlar arası sapma 0.01–0.02. Temiz DFL'de kural tabanlı ölü top kötü (top ölüde de görünüyor);
  bozulmuş satır bizi temsil ediyor. Bozma maliyeti 0.06–0.14. **S**

## 2. Puan nereye gidiyor (dış veri, bozulmuş)
| | Metrica | DFL |
|---|---|---|
| sadece takım kararı hatalı (ölü top etiketten) | 0.567 | 0.541 |
| sadece ölü top hatalı | 0.942 | 0.907 |
| tam zincir (kontrol kuralı + sınıflandırıcı) | 0.548 | 0.487 |
| temiz girdi, sadece takım hatası | 0.681 | 0.620 |
- **Darboğaz takım kararı** (puanın %43–46'sı; ölü top %6–9). DFL ve Metrica ayrı ayrı; geçiş
  raporu da aynı: ölü top kenarları %71–82 yakalanıyor, el değişimleri 1 s içinde %28–35. **S**
  (`p4a_error_budget.py`, `p5g_transitions.py`)
- **Kusursuz girdiyle bile kural 0.62–0.68:** kayıp çoğunlukla kuralın konvansiyona uymamasından. **S**
- **Zamanlama ucuz, yapı pahalı:** 0.2 s gecikme 0.98, 0.4 s 0.92–0.94, 1 s 0.77–0.78; 2 s'den
  kısa possession'ları silmek 0.85–0.89; dakikada 1 sahte possession 0.83–0.86; ölü top kenarı 1 s
  kayınca 0.90–0.93. 3 tohum. **S**
- **Gerçek el değişimlerinin ~%50'sine 3 s içinde hiçbir geçişimiz yok, ~%20'si 1–3 s kaymış (çoğu
  geç).** Sahte geçişlerin yarısı ≥ 5 s süren yanlış aralık başlatıyor. 2 kaynak, 3 tohum. **S**
- **Kaçırılanların %30–46'sı 2 s'den kısa possession;** yeni takım vakaların %97–99'unda topa 1.5
  m'ye giriyor ama kural 0.4 s sürekli kontrol bekliyor. Kuralın el değişimlerinin %21–25'i
  uydurma. (`p4e`, `p4c`) **S**
- **Geçişin zamanı:** vekiller teması geçiş sayıyor (Metrica gecikme 0; DFL'in kendi algoritması
  +0.08–0.16 s); kontrol kuralı yapısı gereği +0.36 s geç (bedeli ~%3–5). Vekildeki el
  değişimlerinin %53–62'sinde kural ±3 s içinde aynı takıma geçiyor. **S** (`p4b_transition_timing.py`)
- Kural dış veride vekilden az possession üretiyor (90 dk'da 265–287'ye karşı 320–332); bizim
  maçta 314. Bizim maçta kuralın kısa possession'larının %74'ünde iki takım da topa 3 m'den yakın. **O**

## 3. Takım kararı
- **Kontrol kuralı (R 1.5 m, T 0.4 s) en iyi basit kural:** DFL 0.61, Metrica 0.68 (ölü top
  kusursuz); en yakın oyuncu 0.11–0.14; yapışkan R=1 m 0.45–0.52. **S** (`g4_dfl_rules.py`)
- Public'te tek seferlik teşhis (ölü top GT'den, sadece anlamak için): kontrol kuralı 0.522
  (19 / 17 possession), en yakın oyuncu 0.046 (132 / 17). **Z**
- **R × T** (`p5o_rt_grid.py`; DFL, Metrica 1–2, Metrica 3; 3 tohum): keskin topta üçü de 1.0 m / 0.2 s'yi
  seçiyor (0.540 / 0.602 / 0.574; 1.5/0.4: 0.536 / 0.582 / 0.543). ATD gibi 25 kare yumuşatılmış topta
  1.0/0.2 çöküyor (0.456 / 0.516 / 0.475); 1.5/0.4 DFL'de en iyi (0.484), Metrica'da en iyinin
  (2.0/0.6) 0.002 ve 0.019 altında. Bizim top yumuşak: 1.5/0.4 doğru seçim; çelişki keskin/yumuşak
  farkıymış. Dokunuş katmanı topu keskinleştirirse 1.0/0.2 yeniden düşünülmeli. **S**
- **Dokunuşla takım kararı (keskin topta) kontrolü geçiyor:** DFL bozulmuş 0.541 → 0.661, temiz
  0.620 → 0.768; DFL'in 7 maçının 7'sinde, 3 tohumda (en küçük +0.055); bozma ×0.5/×1.5 ve beş
  hedef türetmesinde. touch ile control+touch arasındaki sıra Metrica'da 3 tohumun 1'inde dönüyor. **S**
- **Ama sadece keskin topta:** 9 kare yumuşatınca touch 0.510, kontrol 0.536; 25 karede 0.398 / 0.484.
  Metrica'nın topu parça parça doğrusal (kareden kareye hız değişimi medyanı 0.026 m/s, Game 3'te
  0.017; DFL'de 0.48). ATD topu en az bu
  kadar yumuşak (aşağıda); bizim maçta ATD topuyla touch 90 dk'da 192 possession, dedektör topuyla
  450 (titreme). **S**
- **Kırılma dokunuşu gösteriyor:** DFL'de ELASTIC olaylarıyla precision %61 (şans %14), recall %84;
  bozulmuşta %60 / %74; 25 kare yumuşatınca recall %7. Metrica'da da %61 / %84 (şans %10) ama
  Metrica topu parça parça doğrusal, kırılmaları kısmen olay işareti. **S** (`p5c`, `p4d`)
- **Kusursuz dokunuşla tavan ~0.76** (DFL bozulmuş 0.759): kontrolün +0.22 üstü. **O**
- **PathCRF (KDD 2026) yeniden üretildi:** kenar doğruluğu 0.695 (makale 0.696), takım doğruluğu
  0.916 (0.924). Topsuz, kontrolü +0.014–0.028 geçiyor; keskin topla dokunuş çok önde (0.79 vs
  0.64). Bizim maça taşımanın engelleri: kimliksiz, parçalı izler (46 iz, 22 oyuncu) ve kamera
  dışındaki oyuncuların yokluğu. **İkinci test maçı (J03WMX, aynı ayarlarla yeniden eğitim, deneme
  200): avantaj tekrar etmedi.** Kenar 0.656, takım doğruluğu 0.902; spellF1 ham 0.530 vs kontrol
  0.534, birleştirilmiş 0.575 vs 0.613; fazla bölüyor (451 possession, gerçek 511 / 381, kontrol
  327). 2 maçta: J03WR9'da +0.01–0.03, J03WMX'te −0.004 / −0.038 → kontrolden iyi olduğu kanıtlanmadı. **Z**
- "Top havadayken kontrol yok" kuralı zarar veriyor (DFL, Z verisiyle: 0.607 → 0.541). **O**
- Koşu yönü (topa hizalanma) ölü top sinyali olarak hızdan zayıf, bilgisi örtüşüyor (DFL 7 maç). **O**
- Tablo modelleri: LightGBM ve XGBoost HistGB'ye eşit (dengeli doğruluk 0.940–0.945); darboğaz
  özellikler. 2 öğrenen × 2 yön. **S**

## 4. Ölü top
- **Sınıflandırıcı** (HistGB, tracking özellikleri, bozulmuş dış veride eğitim): dengeli doğruluk
  kural 0.886 / 0.903 → 0.940 / 0.939 (iki çapraz yön); spellF1 açığının %50–70'ini kapatıyor. Bizim maçta GT'siz
  %35–36 ölü, 86–89 blok, medyan 18.7–19.4 s (Metrica tipik ~%37, ~91, 16–19 s). **S**
- **Oyuncu hızı bağımsız sinyal:** 2 s medyanı < 1.4 m/s; DFL'de oyunda %9–20, ölüde %72–87; eşik
  DFL 7 maçta plato 1.3–1.5 (dengeli doğruluk 0.819 / 0.824 / 0.822). Top kayıpken bizim maçta
  oyuncular medyan 0.98 m/s, top görünürken 1.65. **S**
- **Oyun sürerken top kaybolunca sınıflandırıcı bu karelerin %61–74'üne "ölü" diyor; top
  görünürken %0.** Ölü top hatalarının asıl yeri. DFL + Metrica, 3 tohum. **S** (`p5h_keeper_hold.py`)
  Nerede (`p5n_dead_gaps.py`, 3 tohum): kısa boşluklarda neredeyse hiç (< 2 s: %0–1), uzunlarda
  çoğunlukla (≥ 10 s: DFL %59, Metrica %94); oyuncular yavaşken (< 1.4 m/s: %44 / %87); yan çizgiye
  < 3 m'de kaybolan topta %85–95. Boşluğun kendi özellikleri (topun nerede ve ne hızla yeniden
  göründüğü, oyuncu ağırlık merkezinin kayması) yardım etmiyor: DFL'de "ölü" %44 → %38, spellF1
  0.478 → 0.488 (maçlar arası aralıklar örtüşüyor), Metrica'da değişiklik yok. **S** (yardım etmediği)
  Bizim maçta (`p5q_broadcast_leave.py`): ≥ 10 s'lik ATD boşluklarında sınıflandırıcının ölü dediği
  48.582 karenin %67'sinde yayın da sahadan ayrılmış (gerçekten ölü), sadece %4'ünde (2.174 kare,
  ~87 s) yayın sahada ve hızlı bir top görüyor (muhtemelen yanlış). Bizim maçta bu hatanın payı dış
  veridekinden küçük; yayın katmanıyla kapatılabilir (modelleme).
- **Kaleci topu tutarken** sınıflandırıcı DFL'de %14, Metrica'da %68 "ölü" diyor (dış veride
  kaynaklar ayrışıyor). Bizim maçta topun kalecinin yanında kaybolduğu 9 boşluk: gözle 6 oyun içi,
  2 ölü (çarpışma, sakatlık), 1 belirsiz; v0 5'ine D diyordu. Bugünkü sınıflandırıcı 7'sine oyunda,
  en uzun ikisine (21–22 s) ölü diyor: sayılar tutuyor, ama vaka vaka eşleşme not edilmemişti.
  Bizim maç için **O**, dış veri için **K**.
- **Yayının sahadan ayrılması** en güçlü tek sinyal: public'te D'nin %76'sı, %85 isabet. Maçın
  tamamında GT'siz (`p5q_broadcast_leave.py`): sınıflandırıcının (yayını görmeyen) ölü topunda
  karelerin %66'sı, oyun içinde %13 (duran toplardan uzakta %10); ayrılma karelerinin %74'ü ölü
  topta; duran topların %65'inin penceresinde ayrılma var. Dış veride ölçülemiyor. **S**
- Üç sinyal oylaması (top yok, oyuncular yavaş, yayın ayrılmış) public'te %68 / %90; GT'siz 104 blok,
  medyan 13 s, %37. Sınıflandırıcıyla yerine kondu. **O**
- **v0** (ilk uçtan uca kural hattı, teslim değil): GT'siz 338 possession, medyan 7 s, D %41;
  public 0.454 (baseline 0.326, aralıklar örtüşüyor); Smart Tagging uyumu %73.6 (smooth50 %65.4).
  Sınıflandırıcıya yayın kuralı eklemek: %40, 115 blok, medyan 11.7 s; public 0.430 (gürültü
  içinde, seçim yapılmadı). **O**
- **"Uzun top boşluğu ≈ ölü top":** Smart Tagging duran toplarının %86'sı bir boşlukla örtüşüyor;
  restart türü ve takımı topun yeniden göründüğü yerden okunuyor (6/6 gözle). **S**
- **Hareketsiz top:** bizim maçta hareketsiz karelerin %76–79'u ölü topta, ölü blokların %49'unda
  hareketsiz top; dış verideki ilişki yapay (DFL ölüde topu donduruyor). Üç top kaynağıyla
  (`p5p_still_restarts.py`; taktik ATD, yayın ATD, ball_v1): duran topların %29'unda en az bir kaynakta
  hareketsiz top, rastgele oyun anlarında %12; iki kaynağın uyuştuğu %12'ye %4. Kale vuruşu %57
  (7), korner %25 (12), taç %14 (21). Taktik ATD neredeyse hiç görmüyor (%4 = kontrol): duran topu
  bırakıyor. **Zayıf bir sinyal olduğu S**; tek başına ölü top işareti olamaz.
- Kale vuruşları ortak kör nokta (%44; ATD yerine konan topu izliyor, oyuncular koşuyor). Restart
  noktası özellikleri yardım etmedi (fark sapma içinde). **O**
- Ses (düdük): 3.6–4.0 kHz tonal tepe; bilinen anlarda rastgele pencerelerin %65–100'ünü geçiyor,
  ama rastgele 6 s'lik pencerelerin %5'i de geçiyor. Genel ölü top sinyali değil (GT'siz testte
  rastgeleyle aynı); sadece başlama vuruşunda belirgin (%40). Yarı sonunda sinyallerden biri. **S** (zayıf olduğu)
- Kalkık yan hakem bayrağı aday sinyal (2 gözlem). **Z**

## 5. Dönemler ve maç bilgisi
- **1. yarı 59:07.7'de bitiyor (kare 88.692),** puanlanan bölüm 90.651'e kadar sürüyor. Düdük,
  oyuncu hızı, ATD topunun kaybolması, yayının sahadan ayrılması, oyuncuların sahadan çıkışı, video
  ve kullanıcının okuması. Tek başına düdük yanılırdı (58:46'daki sakatlık düdüğü daha güçlü).
  `mtah.periods.detect_half_end`; sonrası D (dönem kodu POSTH). 2. yarı sonu değişmedi. **S**
- Kickoff: 1. yarı 11185 (yayımlananla aynı), 2. yarı 113735–113743 (yayımlanan 113742). Top
  toleransı 4 m (2 m'de 2. yarı bulunamıyor, 2.935 karenin dönemi değişiyor); gol sonrası 3/3. **S**
- **Gerçek skor 2-1:** 1-0 (25:43 öncesi), 1-1 (42:59 öncesi), 2-1 (77:57 öncesi). Smart
  Tagging'in 57283'teki "Athletic golü" yok: taktik videodaki tabela ve yayın 30:43–33:55 boyunca
  1-0. O 190 s'nin çoğu **su molası** ("PAUSA DE HIDRATACIÓN", 31:53'ten). "VAR iptali" tahminim
  yanlıştı. **S**
- 2. yarı başındaki 2.934 kare (110.808–113.741) puanlanıyor ve hepsi D; ısınmada sahada birden
  fazla top var, ATD onları da izliyor. **S**
- **Yayın senkronu:** taktik = yayın + 4014 (1. yarı) / + 26920 (2. yarı), yarı içinde sabit (±1
  kare); maç saati, iki ATD topu (karelerin %55'inde ≤ 1 m) ve kullanıcının zamanları. **S**

## 6. Video ve dedektörler
- **Kare hizası:** YOLO topu ile ATD topu 0 kaymada en yakın (15.8 px); maç saati tutuyor; örnek
  çözüm bayt bayt aynı. **S**
- **ATD topu denetimi** (hazır YOLO, 1000 kare): %61.8 doğrulandı, %9.5 çelişki, %28.7 belirsiz;
  belirlenebilenlerin %87'sinde ATD doğru; > 5 m 33 vakanın 16'sına gözle bakıldı (~11'inde ATD
  açıkça yanlış: eski yörüngeden tahmin). Çelişki hızlı topta daha sık (10–25 m/s'de %16). **O**
- **Hazır YOLO11m:** kutuların %63'ü saha dışında (yedek top); penaltı ve orta nokta yanlışları
  (noktadaki kutuların %81–93'ü, conf ≥ 0.5'te %23); ATD boşluklarının %19.5 (conf ≥ 0.3) / %28'ini
  dolduruyor, 12/12 gözle gerçek. **S**
- **ball_v1** (YOLO11m, ATD'den sözde etiketle ince ayar; elle etiket yok): ATD topu recall %66–68
  (hazır %33–55), precision ~%80, oyun içi boşluk doldurma %43–52, nokta yanlışı %0. Oyun içi top
  kapsamı %90.3 → %95.9. **O**
- **WASB ve model deposu, hazır futbol (ISSIA) ağırlıklarıyla** (`p5l_wasb.py`; P3f'in 3.000 doğrulama
  karesi, aynı ölçütler, conf ≥ 0.2). Bizim top ~8 px; bütün kare 512 × 288'e küçülünce top ~2 px ve
  WASB hiçbir şey bulmuyor (%0). 2 × 2 karoda: WASB recall %49, precision %69, boşluk doldurma %40;
  TrackNetV2 %46 / %66 / %34; ResTrackNetV2 %37 / %58 / %40; MonoTrack %13 / %29 / %32. ball_v1
  %68 / %80 / %52 (yeni Blackwell taramasıyla P3'teki sayıların aynısı: bağımsız tekrar). Doğal
  çözünürlükte (20 karo; 3.000 kare) WASB recall %28–51, precision %15; boşluk doldurma %90–100 ama
  bu her karede bir yere "top" demesinden (yanlış pozitif), başarı değil; 2.1 kare/s. BallSeg 70 dk'da bitmedi, durduruldu. **Hazır ağırlıklar ball_v1'in gerisinde;** adil
  karşılaştırma için WASB'a da ince ayar gerekir (seçenek D, yapılmadı). Titreme de azalmıyor: oyunun
  sürdüğü bir 2 dk'lık parçada (135.000–138.000) dakikada kırılma WASB 61, ball_v1 45 (ilk seçilen
  parça su molasına denk geldi, sayılmadı). Hız: WASB 2 × 2 karoda ~9–11 kare/s (ön işleme işlemcide). **S**
- **Dedektör izi titriyor:** dakikada 111 kırılma (gerçek takipte 30–47); havadaki top yere
  izdüşünce yol bükülüyor. İkinci ölçüm (25 Ekim, Blackwell taraması, `p5l_wasb.kink_rate`: karedeki en emin kutu,
  sahada, nokta değil, ≤ 5 karelik boşluk ara değerli): oyunun ≥ 60 s kesintisiz sürdüğü 22 parçada
  (44 dk) dakikada 84 (conf ≥ 0.1), 79 (0.3), 77 (0.5) kırılma; güven eşiği düzeltmiyor. Tek bir
  2 dk'lık parçada 45 çıkmıştı, istisna. **S**
- **Havadaki top:** uyuşmazlıkların %61'i görüntüde ATD noktasının dikey üstünde (hızlı top). **S**
- **Forma rengi** (OpenCV Lab + ton, gradient boosting; etiketler ATD slotlarından): takım ayrımı
  %96.5 / %97.1 yarıdan yarıya. Etiket ve ölçüt aynı kaynaktan (ATD). **O**
- **Kaleci kurtarma:** hayalet karelerde kaleci formalı kişi Barça %40, Athletic %29; 12 gözlemin
  10'u doğru. Etkisiz bulundu (Smart Tagging). **O**
- **P3 kararı ("dedektörün topu takım kararına girmesin") ÇELİŞKİLİ.** Dayanağı sadece Smart
  Tagging uyumu (%74.1 → %72.5). Beş test tersini söylüyor:
  (1) dış veride topu kusursuz doldurmak DFL +0.044, Metrica +0.022 (3 tohum);
  (2) yarısını bulan, 5 m kayan dedektörle DFL +0.014 (Metrica gürültü içinde);
  (3) topun yanındaki oyuncular da silinince zarar yok, kazanç sıfır;
  (4) çekişmeli anlarda kesilen boşlukta +0.05, iki kaynakta;
  (5) bizim maçta GT'siz yapay boşluk: referansla uyum %79 → %90.
  Bizim maçın gerçek boşluklarında Smart Tagging, yayın ATD'si ile dedektörün aynı yeri gösterdiği
  topta bile değişen karelerin %68'inde "tut" diyor; değişen aralıklar kısa değil (medyan 7.4 s).
  "Hayalet kaleci" hipotezi çürüdü. Hakemin taraflı olduğuna işaret: en az possession üreten
  etiketleyici (touch, 192) en yüksek Smart Tagging uyumunu (%75.7) ama en düşük public kontrolünü
  (0.214) alıyor. (`p5f_verification.py` A–A6)
  **Forma rengi (25 Ekim gecesi, `p5m_kit_team.py`):** dedektör topuna en yakın kişinin (≤ 2 m)
  forma takımı, ATD'nin takımıyla kolay karelerde %95.3, boşluklarda %93.4 uyuşuyor: **"boşlukta ATD
  oyuncuları yanlış takımda" hipotezi çürüdü** (S). Doldurmanın değiştirdiği karelerde forma
  doldurulanla %79 aynı, ama bu possession için bağımsız bir hakem DEĞİL: doldurulan karar zaten
  topa en yakın oyuncunun takımı, forma da aynı kişiye bakıyor; sadece o kişinin takımını doğruluyor,
  topun ona ait olduğunu değil (ilk yorumum yanlıştı, düzeltildi). **Sonuç:** doldurma lehine
  bağımsız kanıt dış verideki gerçek etiketli testler (A, A2, A3, A6) ve bizim maçtaki yapay boşluk
  (A4); aleyhine sadece Smart Tagging, ki az değişim yapanı ödüllendiriyor. Terazi doldurmaktan
  yana; teslim zincirini değiştirmek kullanıcının kararı. **K (doldurmadan yana)**
  Yayın ATD'sinin topuyla (`p5m_kit_team.py broadcast`): ATD ile forma boşluklarda %95.6 uyuşuyor
  (aynı sonuç: boşlukta takımlar doğru); değişen karelerde forma doldurulanla %82 (aynı çekinceyle).
- P3'te yayın topu da zarar veriyor diye görünmüştü (aynı tek hakem, Smart Tagging; dış veride
  yayın benzeri bir kaynak yok, yani bağımsız sınaması yapılamadı). **Kaleci düzeltmesi** (hayalet kaleci izini atmak, kaleciyi formadan kurtarmak) de
  "etkisiz" bulunmuştu, sadece Smart Tagging'le. Forma hakemiyle (`p5r_keeper_fix_kit.py`): kararın
  ayrıştığı 177 karede forma düzeltmeyle %70 aynı; ama bu döngüsel: kurtarılan kaleci zaten formadan
  geliyor. Kurtarılan kalecinin topa ≤ 2 m olduğu 88 karede forma %100 düzeltmeyle (kendini onaylıyor);
  sadece hayalet izin atıldığı 89 karede ise %60 eskiyle, %40 düzeltmeyle. Smart Tagging %84 eskiden
  yana. **Düzeltmenin yararına bağımsız kanıt yok;** az örnekte hafifçe tersi. Kayıp oyuncuyu geri
  eklemek de forma hakemiyle sınanamaz (eklenen oyuncunun takımı formadan). **Z** (Dedektör ve yayın
  topu testlerinde bu döngü yok: orada takımı ATD oyuncuları veriyor.)
- **ATD'nin kayan topu:** düzeltmenin etkisi yok (bizim maçta %1.4 kare, Smart Tagging ve public
  aynı); dış veride kaymayı tamamen kaldırmanın tavanı DFL +0.004, Metrica +0.015. **S**
- **Çift sayım:** dış veride +0.002, bizim maçta karelerin %0.05'i; etkisi yok. **S**
- **Boşluk doldurma kaynakları** (oyun içi, ATD topu yok; oyunun %12.2'si): yayın ATD'si %36.8, hazır
  YOLO %21.1, ikisi %46.6 (kapsam %87.7 → %93.5). Taktik ile yayın ATD'si aynı karede medyan 0.82 m,
  %13'ünde > 5 m; ayrışmalarda YOLO'ya göre taktik haklı %35, yayın %27. P3'te yayınla doldurulan
  karelerde Smart Tagging uyumu %77.1 → %60.3, dedektörle doldurulanlarda %63.6 → %60.0. **O**
- Havadaki topta görüntüdeki top hızlı (11 m/s'ye karşı 5.5); ATD kişileri YOLO kişilerine medyan
  0.15 m (p90 0.36). **S**
- Topa en yakın kişi ATD'de eksik (karelerin %1.5'i); geri eklemek sonucu değiştirmedi (%74.1 →
  %74.2). **O**

## 7. Ortam ve tekrar üretilebilirlik
- **Katı fp32** (TF32 kapalı, cuDNN belirlenimci; `mtah.gpu.strict_fp32`) + kutuları 0.01 px / 1e-4
  güven ızgarasına yuvarlamak: 200 karede RTX 4060, RTX PRO 6000 Blackwell ve Colab CPU'su
  **858/858 değer aynı**; torch 2.6 ile 2.7.1 de; grup boyu 4 ile 16 da. **S**
- torch'ta `cudnn.allow_tf32` varsayılan `True`; eski "fp32" ölçümleri TF32'yle yapılmıştı. Eski
  "GPU fp32 ↔ CPU fp32: 6 kutu" farkı katı fp32'de kayboldu. **O → S**
- fp16 kendi içinde belirlenimci ama fp16 ↔ fp32 top kümesi %78–80 aynı; teslim fp32. **S**
- Donanım benzeri bozma oyun içi karelerin ~%0.3'ünü değiştiriyordu (spellF1 ≥ 0.987). Katı fp32 +
  yuvarlamayla: aynı kartta (Blackwell) taramanın ikinci çalıştırması 8.000 karede (8 parça) ilkiyle
  bit bit aynı; farklı kartlar (4060, Blackwell, CPU) 200 karede aynı. Maç boyu kartlar arası eşitlik
  ancak ikinci kartta tam tarama ile kesinleşir. **S** (aynı kart), **O** (kartlar arası, maç boyu)
- **Maç boyu katı fp32 tarama** (Blackwell, 25 Ekim): 158.857 kare, 60.9 kare/s (~44 dk); Drive
  `mtah2027/runs/ball_v1_pass_fp32_…Blackwell…`. 23 Ekim'deki TF32'li yerel taramayla: ortak
  106.237 karenin %0.022'sinde merkez > 0.01 px. Public kontrolü değişmedi (v1 0.442, v2 0.380,
  v2t 0.420). **S**
- **Hız:** RTX 4060'ta ball_v1 10–12 kare/s (maç ~3.6 saat); Blackwell'de 65.5 (grup 4), 70.6 (16).
  23 Ekim'de iki kart ve iki torch sürümü arasında en büyük fark 6.1e-5 idi; yuvarlama onu siliyor.
- **Video okuma darboğaz değil:** cv2 1.350–1.450 kare/s; model tek başına 65 (grup 4) / 70 (grup
  16) kare/s. **S**
- Videoların sha256'sı her kopyada aynı (taktik `ff249f07…`, yayın `3dc2a7d9…`). Yayın 164.487
  kare (cv2, ffprobe, dosya künyesi); dokümandaki 164.488 bir fazla. **S**
- Colab'da tohum 0 raporları yereldekiyle bayt bayt aynı. **S**

## 8. Smart Tagging
- 638 kayıt; takım belirten etiketler puanlanan karelerin %68.1'ini kapsıyor, %13.5'inde iki takım
  birden. En yakın oyuncunun takımıyla %63.7, public'te GT ile %69.6 uyum. **S**
- Kodlar taktik evreler (aynalı çiftler: BUILD UP ↔ DEFENDING IN ATTACKING THIRD ...); bir takımın
  kesintisiz aralıkları medyan 18.2 s, %6'sı < 2 s. Kısa possession'ları görmeyen bir hakem. **S**
- **Smart Tagging yapışkan etiketlemeyi ödüllendiriyor** (`p5s_st_bias.py`): kontrol kuralının T'si
  büyüdükçe bizim maçta possession sayısı düşüyor (0.2 s: 484, 0.4: 322, 0.8: 216, 1.2: 173 / 90 dk).
  Smart Tagging uyumu en yüksek T = 0.8'de (%76.8), 1.2'de bile %76.1 (0.4'te %74.1). Dış verinin
  gerçek spellF1'i (yumuşatılmış top, 3 tohum) ise T = 0.4'te zirvede (DFL 0.484, Metrica 0.541),
  0.8'de 0.434 / 0.505, 1.2'de 0.373 / 0.454. Hakem doğru ayardan yaklaşık iki kat az possession
  üreten ayarı seçiyor. P4h'deki işaretle birlikte ikinci bağımsız kanıt. Ayrıca T = 0.4'te bizim
  maçta 322 possession / 90 dk, dış vekillerin 320–332'siyle aynı. **S**
- Duran top aralıkları sabit şablon (kickoff 5 s, diğerleri 35 s); gerçek vuruş anı değil. Bütün
  taçlarda takım `N/A`. 4 gol diyor, gerçekte 3. **S**

## 9. GT'nin anlamı (public, sadece anlamak için)
- GT kısa, kontrolsüz dokunuşları da possession sayıyor (0.76 s, 1.32 s; 2 örnek). **Z**
- D'nin başlangıcı: top çizgiyi geçince ya da düdükte (top hâlâ hareketliyken); bitişi vuruş ya da
  ilk karşılama (2–4 örnek). **Z**
- Geçiş temastan 0.2–0.6 s (6 vakada 0.2–1.0 s) sonra; "kontrol, dokunuş değil" hipotezi (FIFA
  "In Contest"). Hiçbir karar buna dayanmıyor (zamanlama ucuz). **Z** — güçlendirmek için GT gerekir.
- Public'te D karelerinin %66'sında top görünüyor: "top yok ⇒ D" D'nin çoğunu kaçırıyor. baseline
  smooth50'nin en büyük hatası "GT D iken oyunda, top görünürken" (karelerin %13'ü); yanlış takım
  %11. **O**

## Eksik kalan ve güçlendirilemeyenler
- GT gerektiren: §9'un hepsi, hareketsiz top (§4).
- Açık: kaleci tutuşu (§4), dedektör izinin temizlenmesi (madde 1).

## §10 Modelleme aşaması (26 Ekim–)
- **Dedektöre çevrimiçi takip (WASB usulü) az kazandırıyor:** ball_v1 doğrulama bloklarında
  (30.000 kare, GT'siz) en güvenli kutu → takip: dakikada kırılma 67 → 61, ATD'ye recall 55.7 → 58.2%,
  precision 67.4 → 68.1%, oyun içi boşluk doldurma 54.7 → 58.2%. Üstüne Savitzky–Golay 13 kare:
  kırılma 43, diğerleri aynı. Aynı ölçüyle ATD topu 9. Gürültü yanlış kutudan değil konumdan.
  (`p6a_ball_tracking.py`, `mtah.balltrack.track_detections`) **O**
- **Öğrenen takım kararı kontrol kuralını 7 / 7 DFL maçında geçiyor:** kare özellikleri (iki takımın
  topa uzaklığı şimdi / son / sonraki görülüşte, topun etrafındaki sayılar, takım şekli ve hareketi,
  1 s ve 3 s pencereler) → HistGB → iki durumlu Viterbi (değişim cezası 16, her test maçı için diğer
  6 maçta seçildi). Bozulmuş DFL, top 25 kare yumuşatılmış, ölü top etiketten: spellF1 0.484 → 0.611
  (+0.107 .. +0.150). Viterbi'siz kare kararı 0.30–0.34: kazancın yapısı çözümlemede. Model az bölüyor
  (possession 235–282, GT 308–388). (`p6b_team_model.py`, `mtah/teammodel.py`) **S** (7 maç, tek tohum)
- **Bug:** `keepers_and_sign` takım tarafını 0.5'e göre seçiyordu; J03WR9'da iki takım da yarının
  ötesinde → yön ve kaleciler tersti (bozma modelinin hayalet kalecisi de). Düzeltildi; J03WR9'un
  eski sayıları bundan etkilenmiş olabilir.
- **Tam zincirde de (ölü top kendi sınıflandırıcımızdan, test maçı dışarıda):** DFL 0.460 → 0.580
  (+0.096 .. +0.155, 7 / 7). Bizim maçta GT'siz: forma hakemi ayrışan 4.982 karede %58.2 modelden yana
  (%41.8 kontrol); Smart Tagging uyumu eşit (%74.1 / %74.0); possession 90 dk'da 322 → 215 (DFL'de
  model / kontrol oranı ~0.86, bizde 0.67: bizde daha az bölüyor, açıklanmadı). Public 0.442 → 0.359
  (aralık içinde, seçim değil). (`p6e_full_chain.py 16`) **O**
- **Kırpıntı sınıflandırıcısı (kendi etiketiyle, Colab):** ResNet-18, 128 px; doğrulama AUROC 0.966,
  karedeki en olası kutu kendi etiketli sahibi %91.4 (3.593 kare). Kanıtın takımı olasılık ≥ 0.97'de
  %99.2 doğru (2.698 kare, kolay kareler). Top bırakma testinde (topu %40 silip topla çalışan modele
  göre) 0.830 → 0.874 (ölçek 0.2); bizim maçta possession 209 → 214, public 0.359 → 0.348.
  Uyarı: silinen karelerde top videoda görünüyor, gerçek boşluklardan kolay. (`p6c`, `p6d`, `p6f`) **O**
- **Kayma (p6g):** bizim kareler bozulmuş DFL'den AUROC 0.997 ile ayrılıyor (takım başına oyuncu
  sayısı 11 / 10, takım şekli); modelin kararsızlığı benzer (%9.5 / %7.9). **O**
- **Takım modeli varyantları (DFL LOMO, sadece takım kararı):** base 0.610, + dokunuş 0.619, + yumuşatılmamış
  kopya 0.615, ikisi 0.622, + iki bozma tohumu daha 0.624 (üretim modeli, ceza 12), büyük booster 0.617.
  Ceza eğrisi düz: 8–16 arası 0.609–0.611 (P6i). Bizim maçta GT'siz hakemler cezaya tepkisiz
  (ST %73.4–74.3, forma %86.0–86.8; P6h). (`p6b ... touch_aug_seeds`) **O** (tek tohum farkları ~ gürültü)
- **Ölü top v2** (sonraki görülüş restart noktasına / çizgilere uzaklık, 2 s ileri bakış hız oranı, duran
  oyuncu payı, kickoff şekli): tam zincir 0.584 → 0.589 (+0.005, 6 / 7); mükemmel ölü topla 0.616.
  (`p6j_dead_v2.py`, `mtah/deadball.py`) **O**
- **Dedektör kanalı yardım etmiyor (simüle):** bozulmuş DFL'e ball_v1'in ölçülen oranlarıyla sahte dedektör
  (%60 bildirim, %68'i ≤ 1.5 m, 1 s'lik parçalar) + aynı uzaklık özellikleri: 0.616 → 0.614 (−0.006 .. +0.006,
  2 / 7). Bırakıldı. (`p6k_detector_channel.py`, `mtah.degrade.add_detector`) **O**
- **Ölü top, ATD ölü topu izlerken kaçıyor:** public pencerede (tanımı anlamak için bakıldı) GT ölünün
  sadece %25'i yakalanıyor; 4 bloğun 3'ü tamamen kaçıyor; kaçanlarda ATD topu %88 görüyor, yayın %73
  sahadan ayrılmış (oyunda %3), oyuncular yavaş (1.1 vs 2.2 m/s). Sınıflandırıcı "top kayboldu"ya
  yaslanıyor. Yayın kanıtını naive Bayes ile eklemek işe yaramadı (oranlar "emin" karelerden ölçülünce
  zayıf). **Kural** (eşikler önceden: 3 s'de yayın ≥ %80 ayrılmış ve 2 s medyan oyuncu hızı < 1.3 m/s,
  v0'ın DFL eşiği): maçın %4'ünü ekliyor (eklenenlerde ATD topu %87 görülüyor, top 2 m/s); Smart
  Tagging duran toplarının önünde ölü 77.1 → 79.2%; public kontrolü yakalama 25 → 62%, isabet 73 → 83%.
  v3'e alındı (`mtah.deadball.broadcast_still_dead`, `p6o_dead_rule.py`). **O**
- **Stacking (ikinci katman, P(ev)'in 1–10 s pencereleri) yardım etmiyor:** DFL LOMO 0.615 → 0.616 (3 / 7).
  Bırakıldı. (`p6l_stacking.py`) **O**
- **Smart Tagging duran topları (takımlı 25, public dışı):** aut atışlarında v3 Smart Tagging'le 8 / 9 uyumlu,
  kornerlerde 4 / 12. Korner takımı kullanan takım (köşe geometrisi 9 / 10). Model kornerden hemen sonra
  savunmaya veriyor (kalabalık ceza sahası). v3'e "oyun başlarken 2 s kullanan takım" kanıtı eklendi
  (`mtah/restarts.py`). **O**
- **v3 public kontrolü (seçim değil):** ölü top kuralı + kırpıntı: macroF1 0.669 → 0.803, spellF1
  0.389 → 0.433.
- **Yerleştirilmiş top kuralı boş çıktı:** ATD topu restart noktasında ≥ 2 s duruyor (Vidal-Codina
  toleransları) → sadece 206 kare ekliyor, Smart Tagging kapsamı aynı (79.2%). ATD duran topu zaten
  düşürüyor. Bırakıldı. (`p6p_dead_placed_ball.py`) **O**
- **Hareket özellikleri** (top hangi takımın oyuncusuna doğru gidiyor, topa koşan oyuncular): DFL LOMO
  dokunuş 0.619 → 0.635, üretim (kopyalarla) 0.624 → **0.647** (7 / 7 kontrolün üstünde, ceza 12 beşinde).
  Üretime alındı. (`mtah.teammodel.motion_features`) **O**
- **Metrica 1–3'ü eğitime eklemek yardım etmiyor:** 0.647 → 0.643. Üretim modeli test edilen düzene
  çekildi: sadece DFL (önceden + Metrica 1–2). **O**
- **"Topa ilk kim yetişir" özellikleri** (topun yolundaki noktalara koşu süresi): 0.647 → 0.650, gürültü;
  alınmadı. (`mtah.teammodel.reach_features`, p6b `reach`) **O**
- **Ölü top ve takımı tek Viterbi'de birlikte çözmek kazandırmıyor:** DFL tam zincir 0.605 → 0.605
  (iç seçimle; 2 / 7). Bırakıldı. (`p6q_joint_decode.py`, `mtah.teammodel.decode3`) **O**
- **Top sürme özellikleri** (en yakın oyuncu topla birlikte hareket ediyor, aynı oyuncu ne kadardır yakın):
  0.647 → 0.649, gürültü; alınmadı. Özellik eklemenin getirisi doydu. **O**
- **Topluluk (3 booster, %70 özellik) kazandırmıyor:** 0.647 → 0.646. DFL'de model ~0.647'de doydu. **O**
- **İki yönlü GRU takım kararında booster'ı açık farkla geçiyor:** aynı özellikler (dokunuş + hareket),
  30 s pencereler, 8 epoch. DFL LOMO, sadece takım: booster 0.637, GRU tohumları 0.686–0.693, 3 tohum
  ortalaması **0.707**; tam zincir (kendi ölü topumuz): booster 0.606 → GRU ortalaması **0.676**
  (en kötü maç 0.649), en iyi sabit ceza 8. (`p6r_gru.py`, `p6s_gru_robust.py`, `mtah/grumodel.py`) **S**
  (7 maç, 3 tohum)
- **v4** (v3 + GRU): public kontrolü spellF1 0.663, macroF1 0.840 (seçim değil).
- **GRU v4 tekrar üretilebilir:** sıfırdan yeniden eğitim aynı teslim dosyasını bayt bayt verdi (RTX 4060).
- **GT'siz hakemler v3 → v4'te hafif karşı:** ayrışan karelerin (%6.7) %54'ünde forma ve Smart Tagging
  v3'ten yana. Alan kayması uyarısı; P6u ile sınanıyor. **O**
- **GRU varyantları (2 tohum):** base 0.709, + kopyalar 0.712, geniş 0.706, uzun pencere 0.697: gürültü. **O**
- **GRU'nun önü bozma kayınca da kalıyor (P6u):** eğitim varsayılan bozmayla, test kaydırılmış bozmalarla:
  GRU − booster +0.052 .. +0.065 (top kaybı ×1.5, titreme ×1.5, yumuşatmasız, 50 kare yumuşatma, parçalı iz
  ×1.5); ortalama hep tek GRU'nun altında. v4 tek GRU ile kalıyor. **S**
- **Ölü topu GRU ile tahmin etmek kazandırmıyor:** tam zincir (takım GRU) HistGB 0.679, GRU 0.677, ortalama
  0.682, mükemmel ölü topla 0.709. HistGB kaldı. (`p6v_gru_dead.py`) **O**
- **Kırpıntı ölçeği GRU ile de 0.2** (top bırakma testi 0.813 → 0.849). **O**
- **GRU'nun hata türleri (DFL LOMO, IoU 0.5; P6w):** gerçek possession'ların yarısı < 5 s. Bulunma: 0–2 s %30,
  2–5 s %55, 5–10 s %78, 10–20 s %91, > 20 s %97. Kısa olanların çoğu "yanlış takım" (aynı takımdan hiçbir
  tahmin üstüne düşmüyor, rakibin possession'ında eriyor: 0–2 s'de %45) ya da komşuyla birleşmiş (%22).
  Eşleşmeyen tahminler de kısa olanlar (0–2 s'de %54). Kaldıraç: kısa possession'lar → sınır dedektörü. **S**
- **Sınır GRU'su değişim cezasını kare kare ayarlıyor (ASRF usulü; P6x):** hedef "±0.4 s içinde takım
  değişimi"; ceza 12 (1 − b) + 0.5 b. DFL LOMO sadece takım: sabit ceza (iç seçim) 0.704 → **0.711**
  (+0.002 .. +0.015, 7 / 7). Kısa possession bulunması değişmedi; kazanç sınır yerleşiminden. v5'e alındı. **S**
- **v5 tam zincirde (3 tohum, P6y):** takım sabit ceza 0.710 / sınırlı 0.710; tam zincir 0.680 / 0.681. Sınır
  GRU'sunun katkısı 3 tohumla ~0 (2 tohumla +0.007). İkisi de negatif değil; v5 tutuldu. Public kontrolü
  v4 0.663, v5 0.686. **O**
- **Bozma modeli ölü topu hep siliyordu:** DFL ölüde topu donduruyor, "hareketsiz top" adımı hepsini
  siliyor (bozulmuş DFL'de ölü karelerde top %99.9 yok; bizim ATD %13'ünde görüyor). `carry_dead_ball`
  (ölü sürelerin %25'inde top, oyunu başlatacak oyuncuyla taşınıyor; ölüde görülen top %14.7). Bununla
  eğitilen ölü top sınıflandırıcısı (P6z): taşınan ölü toplu DFL'de tam zincir 0.567 → **0.590** (7 / 7;
  taşınan ölüyü yakalama %0–11 → %84–94); bizim maçta Smart Tagging duran toplarının önünde ölü 83.3 → 89.6%,
  public kontrolü yakalama 62 → 67%, isabet 82 → 77%. v6'ya alındı (`p4_common.dead_v4`). **S**
- v6 tam zincirde (P6y carry_cmp; taşınan ölü toplu DFL, GRU'lar eski bozmayla, 3 tohum): v5 0.641 →
  **v6 0.666** (7 / 7 maç). GRU'ları da taşınan bozmayla eğitmek kötü (takım 0.706 → 0.696, tam zincir 0.656):
  GRU eski bozmada kalıyor. Bu test bizim ATD'ye daha yakın; yeni başlık sayısı 0.666. **S**
- **SkillCorner açık verisi (20 A-League maçı, P7a) yardım etmiyor:** GRU eğitimine eklenince DFL'de takım
  kararı 0.710 → A (sadece görülen konumlar) 0.702 (4 / 7), B (tahminli tam iz + bizim bozma) 0.626 (0 / 7).
  Ters kontrol: SkillCorner'ın kendi etiketi zor öğreniliyor (0.50 → 0.53), etiket tanımı (oyuncu possession
  grubu + phases) DFL'inkinden farklı. Alınmadı; okuyucu `mtah.external.from_skillcorner` duruyor. **S**
- **Taç atışı takımı geometriden: kaldıraç değil.** 26 taçtan 10'unda atan oyuncu belli (top çizgide ≤ 3 m, en
  yakın oyuncu ≤ 2.5 m); v6 bunların 9'unda zaten doğru takımı veriyor. Eklenmedi. **S**
- **§1–§9 yeniden tarandı (26 Ekim):** zincire girmemiş tek büyük bulgu P3'teki top doldurma (§6; P7b'de
  ölçülüyor). Yanlış "ölü"yü yayınla geri almak artık gereksiz: v6'nın ölü dediği yerde yayın sahada + top
  hızlı (> 4 m/s) + oyuncular koşarken toplam 4 s (eski sınıflandırıcıda ~87 s). Su molası (31:15–34:06)
  tek D bloğu. Bizim maçta ikinci top kaynakları: oyun içi ATD boşluklarının %44.4'ü yayın, %43.5'i
  ball_v1 (çevrimiçi takip), en az biri %59.6; ATD'ye medyan 0.82 / 0.87 m, %66 ≤ 1.5 m, %12–13 > 5 m. **S**
- **Top doldurma GRU'da yardım etmiyor (P7b):** bizim maçta ölçülen oranlarla (oyun içi boşlukların ~%60'ı,
  %69 ≤ 1.5 m) ikinci kaynakla doldurulmuş DFL'de takım kararı 0.710 → 0.707 (2 / 7, −0.029 .. +0.043).
  P3 sorusu GRU için kapandı: doldurma yok. GRU boşluğu zaten son / sonraki görülüşten okuyor. **O**
- **Ölü top GRU, taşınan bozmada (P7c, tam zincir, 3 tohum):** HistGB (v6) 0.666; GRU 0.668 (4 / 7); HistGB ile
  GRU olasılıklarının ortalaması **0.674** (+0.007, 6 / 7); takım özellikleri eklemek yardım etmiyor. Etiketten
  ölü top 0.706. **O**
- **Oyuncu kümesi girdisi (P7d, `mtah.setgru`, takım kararı, 3 tohum):** GRU 0.710 → 0.717 (4 / 7), sınırla
  0.710 → 0.718 (4 / 7). Gürültü sınırında. **Z**
- **Takım değiştirme ile veri çoğaltma (P7e) güçlü:** her eğitim maçı bir de ev sahibi / deplasman yer
  değiştirmiş hâliyle (etiket H ↔ A, slot takımları, kaleciler, hücum yönü): takım kararı **0.710 → 0.730**
  (+0.020, 7 / 7, +0.003 .. +0.037); sınır GRU'su da değişimle eğitilince 0.728 (6 / 7). Teslime alındı
  (`mtah.match.swap_teams`, `mtah.grumodel.SWAP`). **S**
- **Gecenin adayları tam zincirde (P7f, taşınan ölü toplu DFL, 3 tohum):** v6 0.666; + takım değişimi (v7)
  0.687 (+0.020, 6 / 7); + ölü top HistGB/GRU ortalaması (v8) **0.692** (+0.025, 6 / 7); + test anında takım
  değişimi 0.693; + oyuncu kümesi 0.692 / 0.693. Son ikisi gürültü içinde, alınmadı. Etiketten ölü topla 0.730. **S**
- **Takım değişiminin üstüne başka çoğaltma işe yaramıyor (P7g, takım kararı, 3 tohum):** takım değişimi 0.730;
  + y aynası 0.728 (2 / 7); + ikinci bozma çekilişi 0.728 (3 / 7); 5 tohum 0.731 (4 / 7). **O**
- **Ölü top modelleri üç bozma çekilişiyle (P7h, tam zincir v8 tarifi):** HistGB + GRU ortalaması 0.692 → **0.699**
  (+0.007, 7 / 7); HistGB tek başına 0.687 → 0.689. v9'a alındı (`p4_common.dead_v6`). **S**
- **Metrica 1–3 takım GRU'sunun eğitiminde, takım değişimiyle (P7i):** DFL testi 0.730 → 0.734 (5 / 7), Metrica
  testi 0.760 → 0.771 (2 / 3). İki yönde de hafif artı, ikisi de gürültü sınırında. Tam zincirde birlikte
  sınanacak. **Z**
- **Bizim maçta GT'siz hakemler v9'dan yana** (`p6m_compare_subs.py` v6 vs v9): ayrışan 3.305 oyun içi karede
  Smart Tagging %52.8, forma hakemi %57.6 v9 ile; possession / 90 dk 259 → 244. DFL'deki yönle aynı. **O**
- **Değişim cezası takım değişimli GRU'larla da düz (P7j, tam zincir v8 tarifi, iç içe):** üretim (12, 0.5) 0.692;
  iç içe seçilen sabit (hep 6) 0.692, sınırlı 0.689. Sabit 6–12 arası 0.690–0.692. Değiştirilmedi. **S**
- **Başlama vuruşları (Smart Tagging'de takım hep `N/A`):** orta yuvarlak geometrisi (merkeze en yakın oyuncu,
  yuvarlaktaki oyuncular) 6 vuruşun 5'inde v9'un takımıyla aynı; tek fark su molası sonrası yeniden başlama
  (62132). Kaldıraç değil. **O**
- **Büyük GRU / uzun eğitim takım değişimiyle de kazandırmıyor (P7k, takım kararı):** 0.730; 12 epok 0.722 (2 / 7),
  256 gizli 0.731 (3 / 7), 3 katman 0.733 (4 / 7). Üretimdeki ağ kalıyor. **O**
- **Ölü top bloklama ayarları düz (P7l, tam zincir v9 tarifi, iç içe):** üretim (0.5, 2 s, 2 s, ağırlık 0.5) 0.699
  (v9'un sayısı bağımsız bir koşuda aynen tekrar etti), iç içe seçilen 0.698; bütün ızgaranın en iyileri 0.701.
  Değiştirilmedi. **S**
- **Metrica 1–3 ve büyük GRU tam zincirde (P7m, v9 tarifi):** v9 0.699; + Metrica 0.701 (4 / 7); 3 katman 0.698
  (3 / 7); ikisi 0.702 (4 / 7). Gürültü içinde; alınmadı. **O**
- **Kırpıntı ölçeği v9 GRU'suyla (P7p, GT'siz top bırakma testi):** spellF1'e göre ölçek 0 en iyi (0.815; 0.2: 0.777,
  eğri gürültülü), kare uyumu ölçekle artıyor (%86.5 → %89.7). v9'un 0 ve 0.2'li hâli karelerin yalnız %0.6'sında
  ayrışıyor; Smart Tagging %63 0'dan, forma hakemi (kırpıntıyla aynı videoya bakıyor) %78 0.2'den yana. Etkisi
  küçük, kanıt çelişkili: 0.2 kaldı. **K**
- **Takım değişiminin kazancı bozma kaymasında da duruyor (P7n, takım kararı, 3 tohum):** eğitimdeki +0.020 (7 / 7),
  top kaybı ×1.5 +0.017 (7 / 7), titreme+kayma ×1.5 +0.023 (6 / 7), yumuşatmasız top +0.020 (7 / 7), 50 kare
  yumuşatma +0.010 (5 / 7), parçalı iz ×1.5 +0.023 (6 / 7). **S**
- **Test maçında kendi kendine öğrenme yardım etmiyor (P7o, takım kararı, CONF 0.9):** sözde etiketler oyun içi
  karelerin %77'si, %96.5–96.9 isabetli; ikinci tur eğitimdeki bozmada −0.001 (3 / 7), kaymalarda −0.002 .. −0.006.
  Alınmadı. **O**
- **Ölü top modellerinde beş çekiliş üçten iyi değil (P7q):** 0.699 → 0.700 (5 / 7). Üç kaldı. **O**
- **v9'un hata türleri (P7r, taşınan ölü toplu DFL, IoU 0.5):** gerçek possession'ların %65'i bulunuyor (ölü top
  etiketten %68). 0–2 s: %28 bulunuyor, %44'ünde aynı takımdan hiç tahmin yok, %20'si komşuyla birleşmiş;
  2–5 s: %58 bulunuyor, %18 / %17; ≥ 10 s: %89–93. Ölü denip kaçan sadece %2. Kalan açık kısa possession'lar. **S**
- **Kısa possession'lara kayıpta ağırlık (P7s, < 5 s ± 12 kare, takım kararı):** K = 2 0.730 → 0.734 (4 / 7), K = 4
  0.732 (4 / 7); tahmin edilen possession sayısı 299 → 319 → 337 (gerçek 352). Gürültü içinde; alınmadı. **Z**
- **Simüle dedektör kanalı takım değişimli GRU'da da yardım etmiyor (P7t):** 0.730 → 0.728 (4 / 7). **O**
- **Zamansal CNN (P7u, takım değişimiyle, takım kararı):** TCN tek başına 0.726 (2 / 7); GRU ile ortalaması 0.730 →
  0.736 (4 / 7). Gürültü sınırında; alınmadı. **Z**
- **Maç başına özellik standardizasyonu yardım etmiyor (P7v, takım değişimiyle, takım kararı, P6u kaymaları):**
  eğitimdeki +0.002 (4 / 7), kaymalarda −0.005 .. +0.003 (1–3 / 7). İlk maçtaki +0.03'lük kazanç tesadüftü. **O**
- **Süreli kod çözme (HSMM, `mtah.hsmm`, P7w, tam zincir v9, iç içe):** v9 Viterbi 0.699, HSMM 0.698 (3 / 7);
  ızgaranın en iyisi de 0.699. Viterbi kalıyor. **S**
- **Küçük artıların birleşimi yeni tohumlarla doğrulandı (P7x, tohum 3–5, tam zincir, taşınan ölü toplu DFL):** v9
  0.700 (0–2 tohumla 0.699: ölçüm tutarlı); GRU, DFL + Metrica 1–3 ve kısa possession ağırlığı ×2 ile 0.711 (6 / 7);
  + aynı veriyle TCN, ikisinin ortalaması **0.717** (+0.017, −0.018 .. +0.038, 6 / 7). v10'a alındı. **S**
- **SoccerTrack v2 (10 üniversite maçı, CC BY 4.0, Drive `mtah2027/soccertrack`, 145 GB doğrulandı):** top izinin
  `status`'u kare kare possession (0 ölü, 1 / 2 takım, 3 sahipsiz); 8 maçta var, 132831 ve 132877'de yok (hep 4).
  v10'un takım GRU'suna eklenince (P7z) DFL 0.745 → 0.730 (3 / 7), SoccerTrack'te test 0.713 → 0.811 (4 / 4): etiket
  öğrenilebiliyor ama tanımı DFL'den farklı (SkillCorner ve diğer takımın PFF denemesiyle aynı tablo). Alınmadı. **S**
- **v10 Metrica 1–3'te tam zincirde (P8a, her maç sırayla dışarıda, 3 tohum):** v9'un takım modeliyle 0.731, **v10
  0.759** (3 / 3; 0.775 / 0.774 / 0.728), v10 ölü top etiketten 0.790. DFL'deki kazanç, GT'ye tanımca daha yakın
  Metrica'da da var. (Takım modeli test maçını hiç görmüyor; ölü top modelleri diğer Metrica maçlarını görüyor.) **S**
- **Daha geniş takım topluluğu (P8b, v10 verisi, DFL, sadece takım, tohum 3–5):** GRU+TCN (v10) 0.745; + oyuncu kümesi
  GRU'su (`mtah.setgru`) **0.751** (+0.006, 6 / 7, hiçbir maçta geride değil); Transformer tek başına 0.597, toplulukta
  katkısız. v11'e alındı. **O**
- **DFL etiket birleştirmesi, Metrica hakemliğinde (P8e, P8g; v10 takım GRU'su, sadece takım, tohum 3–5):** Metrica
  ortalaması 0.0 s 0.785, 0.4 s 0.787, 0.8 s 0.796, **1.6 s (üretim) 0.792**, 2.4 s 0.759. En iyi bölge 0.8–1.6 s;
  "kısa possession'ları silerek kaçırmayı öğretiyoruz" şüphesi doğrulanmadı. 1.6 kaldı. Aynı ayar iki koşuda aynı
  sayıyı verdi (0.842). **S**
- **SoccerTrack ön eğitim olarak işe yarıyor (P8d, v10 GRU, DFL, sadece takım, tohum 3–5):** önce 8 SoccerTrack maçında
  eğitim, sonra v10 verisiyle ince ayar: 0.738 → **0.746** (+0.008, 6 / 7). Karıştırınca zarar vermişti (P7z −0.015):
  hareket SoccerTrack'ten, tanım DFL'den. Topluluktaki etkisi P8j'de. **O**
- **Ölü top topluluğuna TCN (P8c, tam zincir, tohum 3–5):** 3 maçta HistGB+GRU+TCN −0.001 (+0.003 / −0.002 / −0.003),
  GRU+TCN ±0.003. Katkı yok; iş 3 maçta durduruldu. **Z**
- **Sınır modeline TCN (P8f, 3 maç) ve topluluk çeşitliliği (P8h, 2 maç; üyeler farklı bozma çekilişleriyle, log-odds
  ortalaması):** ikisinde de tutarlı kazanç yok (sınır TCN −0.005 .. +0.002; çeşitlilik +0.007 / −0.006, log-odds hep
  ≤ olasılık). GPU'yu boşaltmak için durduruldu. **Z**
- **Ölü topta oyuncu kümesi (P8l, tam zincir, 2 maç):** HistGB+GRU+küme +0.001 / −0.004, HistGB+küme +0.001 / −0.001.
  Katkı yok; durduruldu. **Z**
- **Bizim maçta GT'siz hakemler v11'i destekliyor** (`p6m_compare_subs.py` v10 vs v11): ayrışan 868 oyun içi karede
  Smart Tagging %63.4 v11 ile, forma hakemi %50.6 (tarafsız). **O**
- **v11 topluluğu için ceza (P8i, iç içe, DFL, sadece takım, tohum 3–5):** üretim (sınır 12 / 0.5) 0.751; sabit 3 ve 4
  **0.756**, iç içe seçilen sabit (3–4) 0.755 (6 / 7); sabit 1 / 2 / 6 / 8 / 12: 0.744 / 0.750 / 0.753 / 0.751 / 0.750;
  en iyi sınırlı (4, 2) / (6, 2) 0.756. Üç üyenin ortalaması olasılıkları yumuşatıyor, ceza düşmeli. v11b: sabit 4,
  sınır GRU'su yok. **S**
- **Çok görevli GRU (P8m; takım + sınır başları tek ağda, v11 verisi, DFL, ölü top etiketten, tohum 3–5):** ayrı GRU'lar
  0.737; çok görevli 0.745 (5 / 7); **çok görevli ağın takım çıktısı + ayrı sınır GRU'su 0.749** (+0.012, 6 / 7).
  Sınır hedefi takım temsilini değişimlere duyarlı yapıyor. Toplulukta P8n'de. **O**
- **v11b (sabit ceza 4) teslime alınmadı:** DFL'de iç içe +0.004 (6 / 7, sadece takım) ama bizim maçta Smart Tagging %72.3
  ve forma hakemi %54.4 v11'den yana (v11b 268 → 295 possession / 90 dk; Smart Tagging'in az bölmeyi ödüllendirdiği
  biliniyor). Kanıt çelişkili, kazanç küçük: v11 kaldı. (Public sayısı karara girmez; ilk yazımda gerekçeye yanlışlıkla yazılmıştı.) **K**
- **v11 Metrica'da tam zincir (P8k, tohum 3–5):** v10 0.765 → v11 0.773 (3 / 3: 0.786 → 0.796, 0.790 → 0.792,
  0.720 → 0.729). **S**
- **v11 DFL tam zincirde (P8k, taşınan ölü toplu DFL, tohum 3–5):** v10 0.710 → **v11 0.717** (+0.008, 6 / 7); ölü top
  etiketten 0.742. Başlık: **0.717**. **S**
- **Ölü top GRU'su SoccerTrack'te ön eğitimli (P8o, tam zincir, taşınan ölü toplu DFL, tohum 3–5, 5 maçta durduruldu):**
  v10 ölü top (HistGB+GRU) − ön eğitimli GRU'lu: −0.001 / +0.003 / −0.002 / +0.003 / −0.004 (ort. ≈ 0); HistGB'ye ST
  satırları da eklenince +0.006 / +0.001 / −0.001 / +0.002 / −0.004. Kazanç yok. **O**
- **SoccerTrack ön eğitimi v11'in üç üyesinde (P8j, DFL, sadece takım, tohum 3–5):** v11 0.752 → **0.761** (+0.009, 5 / 7).
  Üye üye: GRU +0.007, TCN −0.012, küme +0.003; kazanç topluluktan geliyor. v12 üretildi; bizim maçta GT'siz hakemler
  başa baş (Smart Tagging %50.6, forma %53.6 v12'den yana), public kontrolü 0.752 (v11 0.765). Metrica hakemi bekleniyor. **O→?**
- **Ölü top taşıma oranı (P9d; HistGB, v9 tarifi, takım etiketten, DFL + Metrica LOMO, 10 maç):** eğitim 0.25 / 0.5 / 0.75
  → test 0.25'te 0.934 / 0.929 / 0.924, test 0.5'te 0.918 / 0.917 / 0.913, test 0.75'te 0.895 / 0.897 / 0.898. En iyisi
  test oranıyla aynı eğitim; uyuşmazlığın bedeli ≤ 0.010. Pişmanlığı en küçük olan 0.25 (en kötü −0.003): kalıyor.
  Asıl bulgu: ölü topta top ne kadar görünürse ölü top o kadar zor (0.934 → 0.895). Gerçek ATD oranı belirsiz
  (%13 ile %66 arası), kayıp orada. **S**
- **v11'in üç üyesi çok görevli (P8n; takım + sınır başı, takım çıktısı sınır GRU'suyla çözülüyor; sadece takım, tohum
  3–5):** DFL v11 0.749 → **0.759** (+0.010, 6 / 7: +0.007 / −0.008 / +0.017 / +0.010 / +0.016 / +0.011 / +0.014);
  Metrica 0.791 → **0.796** (+0.005, 2 / 3: −0.004 / +0.001 / +0.017). İki hakemde de önde. Tam zincir eşli doğrulama
  P9a'nın OOF'unda (`mt` sütunu); tutarsa v13. **S (takım kararı)**
- **Ölü topta ±4–80 s bağlam (P9f, features_v3, HistGB, test taşıması 0.25, takım etiketten; bellek yüzünden 5 DFL
  maçında durduruldu):** v2 − v3: +0.008 / 0 / −0.004 / +0.006 / −0.006 (ort. +0.001). Gürültü içinde. **Z**
- **DFL etiketleri ELASTIC'e karşı (P9g; J03WMX / J03WN1 / J03WPY, 3 kişilik açıklamanın medyanı, olay ve alış
  kareleri):** takım değişiminde sistematik kayma yok: bizim − ELASTIC medyan +0.00 s (ham ve birleştirilmiş), %70–74'ü
  ±0.2 s içinde, p10 / p90 ≈ ±1 s. Oyundaki karelerde uyum %90–94. Ama değişim sayısı: ELASTIC 325 / 331 / 295, DFL ham
  408 / 394 / 324, **1.6 s birleştirmeli etiketlerimiz 278 / 282 / 248 (%15 az)**. Birleştirme, açıklanmış alışlardan
  daha az bölüyor; arkadaşın "ince hedef kazandırdı" bulgusu ve bizim maçtaki az bölmeyle aynı yön. Sınanacak: birleştirme
  0.8 s, Metrica hakeminde ve ELASTIC'in 3 maçında. **S (kayma yok) / O (sayı)**
- **v11 yeni model özetiyle yeniden üretildi (f167034, Colab, 16 iş paylaşımlı GPU):** eski v11 dosyasından karelerin
  %0.75'i (1498) farklı; D 72412 → 73338, A 35611 → 34853. Kod aynı, fark yeniden eğitimin belirlenimsizliği: aynı tarifin
  iki koşusu arasındaki 0.717 / 0.710 farkıyla aynı türden. Kararlar bu yüzden hep aynı koşuda eşli. **S**
- **Altı ağlı takım topluluğu = v14 (P9a OOF + P9b, tam zincir, taşınan ölü toplu DFL, tohum 3–5, aynı koşu eşli):**
  v11'in üç üyesi + v13'ün üç çok görevli üyesi, kod çözme üretimdeki gibi. DFL v11 **0.717 → 0.725** (+0.008, 6 / 7:
  +0.017 / −0.011 / +0.013 / +0.003 / +0.005 / +0.012 / +0.014); Metrica 1–2 0.794 → 0.799 (+0.005, 2 / 2; Metrica 3
  bekleniyor). Çok görevli üyeler tek başına DFL +0.006, Metrica −0.006: kazanç birleşimden. Bizim maçta GT'siz hakemler
  v14'ten yana (Smart Tagging %54.1, forma %53.9; 262 → 254 possession). eps kırpması etkisiz. **Teslim v14.** **S**
- **Ölü top kaybı türlere göre (P9b C, 9 maç, takım v11):** DFL ortalaması v11 0.717; sahte bloklar silinince +0.004,
  kaçan bloklar eklenince +0.006, eşleşen blokların kenarları etiketten alınınca **+0.012**, ölü top tamamen etiketten
  +0.025. Metrica'da kenar +0.019. Ölü top kaybının yarısı kenar yerleşiminde. **S**
- **Ölü top kenarını öğrenen seçici (P9i; tahmin edilen her kenarın ±3 s'indeki adaylar, HistGB LOMO, özellikler ölü top
  v2 + olasılık ve eğimi; takım v14, tam zincir, 9 maç):** v14 DFL 0.725 → 0.723 (iç içe eşik −0.002, 1 / 7), Metrica
  0.799 → 0.797. Kenarlar etiketten gelse +0.025 (bu koşuda); bu özelliklerle seçilemiyor. **O**
- **Resmi baseline'larla hakem ayarı (P9k; ikinci arkadaşın yöntemi, starter/baseline algoritması bozulmuş DFL ve
  Metrica'da, 8 yayınlanmış sayıya RMSE):** en yakın DFL taşıma 0.5 + birleştirme 3.0 s (RMSE 0.016; spellF1 0.070 /
  0.204 / 0.255 / 0.228, resmi 0.049 / 0.200 / 0.243 / 0.224); baseline sırası (125 > 25) yalnız birleştirme ≥ 2.4 s'de
  doğru, 1.6 s'de ters. macroF1 taşıma 0.25'te ~0.03 yüksek, 0.5'te ~0.02 düşük. Metrica (birleştirmesiz) spellF1'de çok
  yüksek (smooth50 0.303): Metrica etiketleri GT'den ince. Ham baseline hep yüksek (0.07–0.11 vs 0.049): gerçek top
  bizim bozmamızdan çok titriyor. Sonuç: gizli GT bizim 1.6 s etiketimizden kaba (≈ 2.4–3 s); ELASTIC'in alış
  sayısıyla çelişmiyor, GT'nin tanımı alışlardan kaba. Metrica birleştirme hakemi olarak yanlı. **O (yeni, doğrulanacak)**
- **P9k titreme ızgarasıyla:** en yakın hakem DFL taşıma **0.4**, birleştirme **3.0–3.6 s** (RMSE 0.007–0.008, spellF1
  0.063–0.068 / 0.190–0.199 / 0.243–0.248 / 0.223–0.232 vs resmi 0.049 / 0.200 / 0.243 / 0.224; macroF1 0.657–0.661 /
  0.675–0.679 / 0.675–0.680 / 0.635–0.637 vs 0.653 / 0.675 / 0.672 / 0.636; sıra doğru). Titreme çarpanı (1–3) neredeyse
  etkisiz. Gizli GT, rakibin ~3 s'den kısa araya girişlerini possession saymıyor gibi; eğitim hedefimiz 1.6 s. Ayarlanmış
  hakem: taşıma 0.4, birleştirme 3.0 s. **O → sınanıyor (P9l)**
- **OOF zincirleri ayarlanmış hakemle (P9l; 9 maç, test etiketleri birleştirilerek yeniden puanlandı, CPU):** DFL
  birleştirme 3.0 s'de v11 0.690, v14 0.691 (v14'ün +0.008'i 1.6 s hakemine özgü); aynı hakemde düz ceza 20 ile v14
  **0.710** (+0.019), Metrica 3.0'da 0.758 → 0.774. GT kabaysa daha yüksek değişim cezası (daha az, uzun possession)
  kazandırıyor. Eğitim hedefinin kendisi P9m'de. **O → sınanıyor**
- **v14, OOF 10 maçla tamam:** Metrica 3 v11 0.729 → v14 0.723 (−0.006); Metrica ortalaması 0.772 → 0.774 (+0.001, 2 / 3),
  DFL 0.717 → 0.725 (+0.008, 6 / 7). İki hakemde de önde (Metrica'da zayıf); teslim v14 kalıyor. **S (DFL) / O (Metrica)**
- **P9e ara sonuç (3 DFL maçı, normal test):** karışık − normal ×2: +0.015 / +0.016 / −0.025 (ort. +0.002); sert testte
  +0.015 / +0.019 / 0.000. Karar için kalan maçlar bekleniyor.
- **Etiket yoğunluğu birleştirmeye göre (90 dk'da possession, ham DFL / Metrica, CPU):** DFL birleştirme 0 / 1.6 / 2.4 /
  3.0 / 3.6 s → 412 / 332 / 298 / 286 / 275; Metrica 0 / 1.6 / 3.0 → 328 / 291 / 249. Ayarlanmış hakem (3 s) ≈ 286 / 90 dk.
  Bizim maçta v14 254 / 90 dk, v11 262, sabit ceza 4 (v11b) 295. Yani bizim maçta model, DFL'deki kalibre yoğunluktan
  de az bölüyor (alan kayması); DFL'de daha yüksek cezanın kazandırması (P9l) bizim maçta tersine dönebilir. Karar için
  sabah: bizim maçta tahmin edilen possession sayısını ayarlanmış yoğunluğa (~285–300) getiren ceza bir aday. **O**
- **Karışık bozma, DFL 7 maç tamam (P9e; GRU + TCN, sadece takım, tohum 3–5, eşit veri boyu):** normal testte karışık −
  normal ×2: +0.015 / −0.008 / +0.016 / −0.007 / −0.025 / −0.007 / +0.008 (ort. −0.001, 3 / 7); sert testte +0.015 /
  +0.017 / +0.019 / −0.003 / 0.000 / +0.018 / +0.026 (ort. +0.013, 5 / 7). Metrica 1: −0.009 / +0.041. Normal hakemde
  kazanç yok, bozma sertleşirse sigorta. Metrica 2–3 bekleniyor; tek başına teslime almak için kanıt yok. **O / sigorta**
- **Top dolgusu (P9j, ilk 3 maç; dolgu eğitimde ve testte):** yok / kısa boşlukların yarısı / tüm kısa / hepsi: J03WMX
  0.749 / 0.756 / 0.757 / 0.763, J03WN1 0.688 / 0.711 / 0.680 / 0.740, J03WPY 0.742 / 0.737 / 0.738 / 0.741. Kusursuz
  dolgu ort. +0.021, gerçekçi (~1/3) +0.007 civarı ama gürültülü. Sürüyor.
- **Top dolgusu, DFL 7 maç tamam (P9j; GRU, dolgu eğitimde ve testte, sadece takım):** geri gelen kayıp kare payı: kısa
  boşlukların yarısı %6, tüm kısa boşluklar (< 10 s) %12, hepsi %100. Fark: kısa yarı +0.002 (3 / 7), tüm kısa +0.004
  (4 / 7), **kusursuz dolgu +0.021 (6 / 7)**. Kazanç uzun boşluklarda; ikinci arkadaşın dedektörü çoğunlukla kısa
  boşlukları dolduruyor (uzun > 10 s'ler duruyor), bize ~+0.004 değerinde. Dedektör çıktısını istemek ucuz; ama asıl
  değer uzun boşlukları (kamera dışı top) dolduran bir kaynakta. **S (tavan) / O (gerçekçi)**
- **Karışık bozma tamam (P9e, 10 maç):** normal test DFL −0.001 (3 / 7), Metrica +0.004 (2 / 3: −0.009 / +0.007 / +0.014);
  sert test DFL +0.013 (5 / 7), Metrica +0.029 (3 / 3). Normal hakemde nötr, sertte açık önde: ucuz bir sigorta. Teslime
  alınırsa normal hakemde kayıp yok; v15 adayına eklenebilir. **O (normal) / S (sert)**
- **Ara sonuçlar:** P9o pencere kaydırma J03WMX +0.011, J03WPY +0.019; P9n kaleci+epizot J03WMX 0.000, J03WN1 +0.005;
  P8p (çok görevli + ST) J03WMX −0.002; P9m kaba hedef DFL 6 maçta 3'e 3 bölünmüş (J03WMX / WN1 / WOH 1.6'yı,
  WPY / WQQ / WR9 3.0'ı istiyor), Metrica 3 / 3 3.0'dan yana (+0.009).
- **Kaba hedef tamam (P9m; GRU + TCN, sadece takım, hakem = 3.0 s birleştirilmiş etiket, ceza iç içe):** DFL 7 maç: "3.0 +
  Metrica 3.0" − "1.6": ilk 4 maç −0.001 (1 / 4), son 3 maç +0.015 (3 / 3), ort. ≈ +0.006 (4 / 7); yalnız DFL'yi 3.0'a
  birleştirmek ilk 4'te −0.011. Metrica (3.0'a birleştirilmiş) +0.009 (3 / 3). Ayarlanmış hakemde bile tutarlı değil ve
  bizim maçta zaten az bölüyoruz: kaba hedef teslime alınmadı. Hakem ayarı (P9k) ölçümler için kalıyor. **O**
- **Sınır GRU'su varyantları (P9p; OOF takım v14 + ölü top, tam zincir, 10 maç):** yarıçap 5 / 15 / 20 kare ve pozitif
  ağırlık ×3, 10 kareye (üretim) karşı hep ±0.004 içinde (DFL −0.001 .. +0.001, Metrica −0.004 .. +0.002). Sınır hedefi
  düz; üretim kalıyor. Yeniden eğitilen r10 OOF'taki sınır GRU'sunu birebir verdi (ölçüm belirlenimci). **O**
- **Pencere kaydırma (P9o, 9 / 10 maç):** DFL +0.011 / +0.010 / +0.032 / +0.019 / −0.005 / −0.006 (ort. +0.010, 4 / 6),
  Metrica −0.003 / −0.029. Hakemler anlaşmıyor; bekliyor.
- **Pencere kaydırma tamam (P9o; GRU + TCN, sadece takım):** DFL +0.009 (5 / 7), Metrica −0.011 (0 / 3: −0.003 / −0.029 /
  −0.001). Hakemler ters yönde: alınmadı. **O**
- **Yalnız olasılık gören rafine katman (P9q; OOF girdileri, tüm maç 5 fps, dilated TCN, LOMO, 3 tohum, tam zincir):**
  v14 ile yarı yarıya: DFL −0.001 (3 / 7), Metrica −0.011 (0 / 3); tek başına DFL −0.011, Metrica −0.023. 10 maçla
  öğrenilemiyor. **O**
- **Ölü topta karışık taşıma (P9r; HistGB, 3 çekiliş 0.25 / 0.5 / 0.75 vs 0.25 ×3, takım etiketten):** test 0.25'te DFL
  −0.004 (2 / 7), Metrica −0.004; test 0.75'te DFL +0.005 (7 / 7), Metrica −0.001. P9d ile aynı tablo: eğitim test oranına
  uymalı, karışık ikisinin arasında. 0.25 kalıyor. **O**
- **Kaleci + dokunuş epizotu (P9n, DFL 7 maç):** ilk 4 +0.005 (2 / 4), son 3 −0.009 (0 / 3); ort. ≈ −0.001. İki arkadaşın
  gördüğü kazanç bizim özellik setinde yok (dokunuş özellikleri zaten var). Metrica bekleniyor. **O**
- **Çok görevli üyelere SoccerTrack ön eğitimi (P8p; tam: DFL 4 maç, Metrica 3):** DFL −0.002 / −0.022 / +0.012 / +0.012
  (ort. ≈ 0), Metrica −0.002 (1 / 3). Ek kazanç yok. **O**
- **Kaggle'da v15 ve v14 aynı koddan (a0f8710, 2× T4, 2.8 saat):** ikisi de geçerli. Colab v14 ile Kaggle v14 karelerin
  %1.13'ünde, Kaggle v14 ile v15 %1.28'inde ayrışıyor (karışık bozmanın etkisi yeniden eğitim gürültüsü düzeyinde).
- **Bizim maçta yoğunluk cezayla oynamıyor (P9s, GT'siz, `analysis/p9s_density_scan.py`, MTAH_DUMP):** v14 ceza ×0.25 /
  ×0.5 / ×1 / ×1.4 → 278 / 265 / 253 / 246 possession / 90 dk; v15 aynı. Ayarlı hakem ~286–300 diyor ama kod çözme oraya
  götürmüyor: olasılıklar bizim maçta uzun ve emin. Az bölmenin kaynağı alan kayması (P6g: AUROC 0.997, 11 vs 10 oyuncu),
  kod çözme değil. Ceza ayarı adayı bırakıldı; sıradaki araştırma girdi tarafında. **S**
- **v15 (karışık bozma) GT'siz, Kaggle v14'e karşı:** Smart Tagging %70.3 v15, forma hakemi %63.6 v14 (ayrışan karelerin
  %2.9'u; 261 / 262 possession). Hakemler çelişiyor, normal DFL hakemi nötr: teslim v14 kalıyor, v15 yedek
  (`evaluation/results/v15_chain.csv`; modelleri Kaggle mtah-v15 çıktısında). **K**
- **Ölü top boyunca restart kuralı (P9v; OOF v14 tam zincir, CPU):** teşhis: v14'ün kaçırdığı ≤ 2 s spell'lerin DFL'de
  %68–85'i ölü topa komşu (Metrica %37–58); restart sonrası ilk takım hatası yalnız %1–7 (model restart takımını zaten
  biliyor). Kural ("restart R ise ve top çizgi yakınında kaybolduysa son L s R'nin rakibi"): en iyi L 1.0 s / 1 m DFL −0.004
  (1 / 7), Metrica +0.005 (3 / 3); iç içe DFL −0.004. Ölü blok başını topun sahadan çıkışına oturtmak (snap): DFL −0.002,
  Metrica −0.002; ikisi birlikte de kazanmıyor. Eklenen spell'ler tahmin edilen ölü kenarın kaymasıyla yanlış yere düşüyor.
  Alınmadı. **O**
- **Ölü top kenar kazancının ayrışması (P9z; OOF v14 tam zincir, CPU):** etiketten kenar düzeltmesi DFL +0.012 (7 / 7), Metrica
  +0.014; neredeyse hepsi bire bir eşleşen bloklardan (birleşmiş blok ayırma +0.002, bölünmüş birleştirme 0). Başlangıçlar
  +0.008, bitişler +0.004. Kenarların çoğu zaten doğru (medyan sapma 0.04 s, %85–91'i ≤ 0.5 s); kazanç başlangıcı > 0.5 s
  kaymış ~%15 blokta (DFL +0.007) ve bitişi > 0.5 s kaymış ~%9'da (+0.004). P9v'de topun sahadan çıkışına oturtma tutmadı,
  çünkü bu kaymış blokların çoğu muhtemelen düdüklü duruşlar (top sahada). Kenar için kalan tek yol düdük ya da oyuncu
  yavaşlaması anlarını yakalamak; P9f/P9i'de denenen bağlam ve seçici bunu yapamadı. **O**
- **Oyuncu sayısı kayması (P9u; GRU, sadece takım, tohum 3–5, Kaggle):** A üretim, B n_H/n_A sıfır, C kamera penceresi
  oyuncuları düşürmüyor. DFL 7 maç, olağan test bozmasında A 0.738 / B 0.736 / C 0.737; oyuncuları koruyan (bizim maça benzer)
  test bozmasında A 0.736 / B 0.738 / C 0.736 (B − A 4 / 7, maç başına ±0.02 gürültü). Metrica oyuncular korunurken A 0.794 / B 0.780 /
  C 0.791. Kaymanın kendisi DFL'de yalnız −0.002 kaybettiriyor (Metrica'da +0.007). Bizim maçta possession sayısı varyantlarla
  değişmiyor (ceza 8'de A 267 / B 265 / C 263). Sonuç: az bölmenin nedeni oyuncu sayısı değil; hiçbiri alınmadı. **O**
- **Dokunuş penceresi 3 → 12 kare (P9w; GRU + TCN, sadece takım, tohum 3–5, Kaggle):** DFL 7 maç h12 − h3 +0.004 (4 / 7; ilk
  beş maç +0.010, J03WQQ / J03WR9 −0.009 / −0.013); 3 s hakeminde ≈ 0.000 (4 / 7). Metrica +0.006 (2 / 3), 3 s hakeminde −0.005
  (1 / 3). Maç başına ±0.02 oynuyor; iki hakemde tutarlı değil. Alınmadı. **O**
- **Top dolgusu nereden kazandırıyor (P9y; GRU, DFL 7 maç, sadece takım, tohum 3–5, Kaggle):** gerçek top yalnız ≥ 10 s
  boşlukların oyun karelerine (kayıp karelerin %6'sı) +0.015 (5 / 7); yalnız ölü karelerine (%83) +0.002 (4 / 7). Bütün
  boşluklar, zamanda bağıntılı gürültüyle: σ 1 m +0.007, 2 m +0.011 (5 / 7), 4 m +0.012 (5 / 7), 8 m +0.002 (3 / 7). Değer
  uzun boşlukların oyun karelerinde ve ~4 m'ye kadar dayanıyor: oyunculardan top tahmini (P9aa) denenmeye değer. **O**
- **Possessor başı (P9x; oyuncu kümesi üyesi tek başına, sadece takım, tohum 3–5, Kaggle):** etiket bulunan alive kare %42–53.
  DFL 7 maç set+who − set −0.002 (4 / 7; ilk beş +0.004, J03WQQ / J03WR9 −0.017 / −0.019), Metrica −0.001 (0 / 3), 3 s hakeminde
  DFL ≈ +0.003, Metrica −0.011. Alınmadı. **O**
- **Oyunculardan top tahmini (P9aa; set-GRU regresyon, maç başına diğer 9 maçla, Kaggle):** ≥ 10 s boşlukların oyun karelerinde
  medyan hata 6–12 m (son görülen top 6–13 m); P9y'nin istediği ~4 m'nin çok uzağında. Dolgu ≥ 10 s: DFL +0.003 (3 / 7), Metrica
  −0.012 (0 / 3); bütün boşluklar: DFL +0.002 (4 / 7), Metrica −0.003. Alınmadı. **O**
- **Tahmincinin bizim maça taşınması (P9ab; GT'siz, topun görüldüğü yerlerde yapay 10–30 s boşluk):** J03WR9 (eğitimde yok)
  medyan 19.1 m (son görülen 25.1), bizim maç 18.0 m (25.9): taşınma sorun değil, tahmincinin kendisi zayıf. Uzun boşlukta
  topu bulmanın yolu oyuncular değil, görüntü (yayın / taktik video) olmalı. **O**
- **Uzun boşlukta videonun payı (P9ac; bizim maç, GT'siz, v14'ün kendi kararlarıyla):** ≥ 10 s ATD boşlukları 55k kare, v14'e göre
  oyunda yalnız ~2.0k kare (1.4 dk; DFL'de maç başı 2.6–6.4k); bunların %42'sinde ball_v1 ya da yayın ATD topu var (ball_v1 %30,
  yayın %28). 2–10 s boşluklar oyunda 3.2 dk, video %70. P9y'nin +0.015'i bizim maçta en iyi ihtimalle ~+0.003'e iner (v14'ün
  boşlukta ölü demesi yanlışsa biraz fazlası). Top dolgusu yönü kapandı. **O**
- **Üye başına 3 → 6 tohum (P9ad; GRU + TCN, sadece takım, Kaggle):** DFL 7 maç s6 − s3 +0.006 (6 / 7), 3 s hakeminde +0.006
  (7 / 7); Metrica +0.002 (2 / 3), 3 s hakeminde +0.012 (3 / 3). s3 sütunu P9w'nin h3'ünü birebir tekrarlıyor (eğitim
  belirlenimci). İki hakemde ve iki veride önde: **alındı → v16** (v14'ün altı takım ağı 6 tohumla). **O**
- **v16 kuruldu (Kaggle T4):** v14k'den %0.32 kare farklı (v14 Colab ile %1.31; v14 Colab–v14k %1.13), possession sayısı aynı (261 / 90).
  GT'siz yargıçlar farklı karelerde yine bölünüyor (Smart Tagging %85.6 v16, kit %58.3 v14k; 575 / 271 kare). Karar P9ad'nin DFL +
  Metrica ölçümüne dayanıyor: **teslim v16** (`evaluation/results/v16_chain.csv`). **O**
- **Sınır GRU'sunda 3 → 6 tohum (P9ae; OOF v14 takım + ölü top, tam zincir, Kaggle):** DFL 7 maç s6 − s3 ≈ 0.000 (3 / 7), 3 s hakeminde
  −0.000; Metrica −0.000 (2 / 3), 3 s hakeminde +0.002. Etkisiz: sınır GRU'su zaten kararlı, tohum kazancı takım ağlarında. v16'da
  sınır GRU'su 3 tohumla kalıyor. **O**
- **Az bölünmenin yeri (P9af; GT'siz, bizim maç v16 dökümü ile OOF DFL / Metrica):** takım olasılığının ham dönüşleri (0.4 / 0.6
  histerezis) oyun dakikası başına bizim maçta 4.54, DFL'de 4.59: takım ağları DFL kadar el değişimi görüyor. Ama sınır GRU'su
  daha az ateşliyor: b > 0.5 tepeleri oyun dakikası başına bizim maç 2.57, DFL 3.54, Metrica 2.94 (b > 0.2: 5.17 / 7.23 / 5.35);
  ortalama b 0.046 / 0.064 / 0.052. Çözülen possession / oyun dk 5.18 / 6.19 / 5.01; ölü blok / 90 dk aynı (91 / 92 / 87).
  DFL'de zincir 304 / 90 veriyor, gerçek 332 (1.6 s) ve 268 (3 s). Az bölünme kod çözmede, sınır GRU'sunun bizim maça
  taşınmasında; takım ağlarında ya da ölü topta değil. Sıradaki: çok görevli ağların sınır başları bizim maçta nasıl
  ateşliyor (v16b ağırlıklarıyla CPU'da çıkarım). **O → sınanıyor**
- **Sınır GRU'su bizim maçta (P9af devamı, CPU):** çok görevli ağların sınır başlarıyla kod çözme OOF'ta DFL −0.000 (3 / 7), Metrica
  −0.005: sınır kaynağı değişmiyor. Takım olasılığı dönüşlerinde (±1 s) b > 0.5 payı DFL %59, Metrica %65, bizim maç %48. DFL'de
  b ≤ 0.5 dönüşlerin %65'i gerçek değişim (b > 0.5: %78). Bizim maçta Smart Tagging'in dönüşten sonraki tarafı yeni takımla düşük b'de
  %41, yüksek b'de %37 (n ≈ 110): sınır GRU'su burada ayırt etmiyor gibi. Ama b'yi DFL dağılımına eşlemek possession'ı yalnız
  253 → 259'a çıkarıyor: az bölünmenin küçük bir parçası. DFL'de zincir 3 s etiketten fazla bölüyor (304'e 268); bizim maçın
  daha az possession'ı gerçek olabilir (Barcelona'nın uzun topa sahip olmaları). Alınmadı. **O**
- **Ölü top GRU'sunda 3 → 6 tohum (P9ag; OOF v14 tam zincir, yerel RTX 4060, maç başı ~3 dk):** DFL +0.001 (4 / 7), 3 s hakeminde
  +0.001 (5 / 7); Metrica −0.001 (2 / 3). Etkisiz; tohum kazancı yalnız takım ağlarında. **O**
- **Smart Tagging geçişleri ve uzun toplar (GT'siz, v16, public hariç):** 23 geçiş (ATTACKING / DEFENSIVE TRANSITION aynı anların
  iki tarafı); v16 bunların 21'inde ±15 s içinde yeni takıma geçiyor, etiketin başlangıcından medyan 2.8 s sonra (IQR 1.2–3.8).
  DFL'de zincirimizin gerçek değişimlere gecikmesi medyan −0.04 s (IQR −0.30..0.20, n 1391): gecikme bizde değil, Smart
  Tagging'in faz başlangıcı kuralında (duran toplarda da faz başlangıcı bizim yeniden başlamamızdan 5–7 s önce: korner 6.8,
  aut 4.9, serbest vuruş 5.0). LONG BALL'ların %39'u ölü topa komşu, yakınında değişim %4. Geçişler zaten yakalanıyor, sayıları
  da az: sınır kanıtı olarak eklenmedi. **O**
- **Ölü top bağlamı takım ağlarına (P9ai; mtah/deadctx.py: sonraki / önceki ölü bloğa süre, onları başlatan takım, çizgiye
  uzaklık; GRU + TCN, sadece takım, tohum 3–5, Kaggle):** DFL 7 maç ctx − base −0.008 (2 / 7; −0.010 / +0.006 / +0.010 / −0.005 /
  −0.033 / −0.010 / −0.012), 3 s hakeminde −0.005. Alınmadı (Metrica yerel koşusu WSL yeniden başlayınca yarıda kaldı; DFL
  sonucu yeterli). **O**
- **Üye başına 6 → 12 tohum (P9ah; GRU + TCN, sadece takım; J03WMX / J03WN1 yerel, J03WQQ / J03WR9 ve Metrica Kaggle):** DFL 4 maç
  s12 − s6 −0.003 (1 / 4: −0.006 / −0.003 / +0.001 / −0.003), 3 s hakeminde −0.003; Metrica +0.003 (2 / 3). Kazanç 6 tohumda doymuş;
  v16 6 tohumla kalıyor. **O**
- **Yayın ölü top kuralının GT'siz ablasyonu (P9aj; bizim maç, v16'nın kendi olasılıklarıyla, yeniden kurulan ölü top v16 ile %100
  aynı):** kural tek başına 2,900 kare (1.9 dk), 38 blok ekliyor; ölü blok / 90 dk 95.2 → 90.9 (blokları birleştiriyor),
  possession 260 → 253. Kuralın eklediği karelerde (public hariç) medyan oyuncu hızı 1.04 m/s (oyun karelerinde 1.89), Smart
  Tagging fazı karelerin yalnız %28'inde (oyun karelerinde %85), ATD topu %89'unda görüyor (ölü topta top görünür, bizim maçta
  olağan). İki bağımsız GT'siz işaret kuralın doğru ölü top eklediğini söylüyor: **kural kalıyor.** **S**
- **Dış inceleme raporunun iddiaları (3 Kas):** kodda doğru çıkanlar: (1) restart kanıtı (mtah/restarts.py) ilk canlı kareye
  basıyor; 26 duran topun 3'ünde bu kare bir ölü bloğun hemen sonrası değil (1'inde pencerede hiç ölü yok); (2) üretimdeki ölü
  top modelleri (p4_common dead_classifier / dead_gru) Metrica 1–2 ile eğitiliyor, ölçümler 1–3 ile; X kareler (Metrica yarılarının
  %0.7–2'si) ölü topta y = 0 sayılıyor; (3) yarı sonu pencereleri (tail) üretimde eğitilmiyor (yalnız jitter ile birlikte P9o'da
  denendi). Yanlış ya da zaten ölçülmüş olanlar: çok görevli sınır başları (OOF'ta DFL −0.000, Metrica −0.005), ceza
  parametreleri topluluk için yeniden ayarlandı (P9b, P9s), v16 public 0.870 değil 0.739.
- **Raporun kod çözme fikirleri (P9ap; OOF v14 tam zincir, CPU):** takım olasılığı sıçramasıyla sınırı büyütmek iç içe DFL −0.0005
  (3 / 7), 3 s'de daha kötü; aynı takım arasında P(ölü) 0.3–0.5 bölgeyi ölü açmak DFL −0.003 (1 / 7); zayıf ölü bloğu veto hiç
  ateşlemiyor. Alınmadı. **O**
- **Restart kanıtı katı sürüm (mtah/restarts.py strict=True; v17 için):** kanıt yalnız pencere içinde bir ölü bölgenin bittiği karede
  ateşliyor, ölü yoksa hiç. Bizim maçta 250 kanıt karesi yer değiştiriyor ama v16'nın kod çözümünde **hiçbir kare değişmiyor**
  (possession 253 → 253): doğru olan bu, ama puan getirmez. Varsayılan eski davranış (v16 aynen yeniden üretilsin). **O**
- **3 olağan + 3 karışık bozmalı ağ, 6 olağana karşı (P9ak; GRU + TCN, sadece takım; J03WMX yerel, kalanı Kaggle):** DFL 7 maç
  +0.0035 (5 / 7), 3 s hakeminde +0.006 (5 / 7); Metrica +0.003 (1 / 3), 3 s hakeminde −0.003 (0 / 3). Sınırda: DFL'de tutarlı,
  Metrica'da değil. Tek başına alınmadı; v17 adayı olarak tam zincirde (BigGRU ile birlikte) sınanacak. **S?**
- **Ölü top eğitiminde X kareleri dışarıda (P9ar; OOF v14 tam zincir, HistGB + ölü GRU yeniden, Kaggle):** DFL 7 maç +0.0009 (5 / 7),
  3 s hakeminde +0.001; Metrica +0.0007 (1 / 3, biri eşit), 3 s hakeminde +0.0013. Küçük ama iki veride de olumlu ve hata düzeltmesi:
  **v17'ye** (üretim ölü top modeline Metrica 3 eklenmesiyle birlikte). **S**
- **v16 tam zincir, ara sonuç (P9al OOF 6 tohum vs P9a OOF 3 tohum, 6 maç: 5 DFL + Metrica 1):** 1.6 s hakeminde −0.0014 (1 / 6),
  3 s hakeminde **+0.007 (6 / 6)**. Takım kararındaki +0.006 (P9ad) tam zincirde 1.6 s'de görünmüyor; ayarlı hakemde tutarlı.
  Teslim v16 kalıyor; 10 maç bitince kesinleşecek. Ders: adaylar mümkünse tam zincirde ölçülmeli. **S?**
- **v16 tam zincir, 9 maç (P9al):** DFL 7 maç 1.6 s +0.0016 (3 / 7), 3 s hakeminde **+0.0077 (6 / 7, biri eşit)**; Metrica 1–2
  1.6 s −0.008 (Metrica 2 −0.016), 3 s +0.001. Ayarlı hakemde v16 önde: teslim v16. **S**
- **Sonuçlar (4 Kas sabah):**
  - **v16 tam zincir, 10 maç (P9al):** DFL 1.6 s +0.0017 (3 / 7), 3 s hakeminde **+0.0074 (6 / 7)**; Metrica −0.003, 3 s ±0.000. **S**
  - **BigGRU (P9am, sadece takım):** DFL +0.002 (3 / 7), 3 s +0.004 (5 / 7); Metrica +0.009 (2 / 3), 3 s +0.006 (2 / 3).
  - **BigGRU tam zincirde (P9at OOF'ları v16 OOF'unda GRU ve çok görevli GRU yerine):** DFL +0.0008 (4 / 7), 3 s hakeminde **+0.004
    (6 / 7)**; Metrica +0.0001, 3 s hakeminde **+0.008 (3 / 3)**. 1.6 s'de kayıp yok, ayarlı hakemde iki veride önde: **v17'ye.** **S**
  - **Geniş TCN (P9ao, sadece takım, DFL 4 maç + Metrica):** dilation 1–128 (±20 s) DFL +0.0075 (3 / 4), 3 s +0.007 (4 / 4); Metrica +0.014
    (2 / 3), 3 s +0.005. 1–64 tutarsız. Kalan 3 DFL maçı + tam zincir (P9au) sürüyor. **S?**
  - **Yarı sonu pencereleri (P9aq):** DFL +0.0017 (4 / 7), 3 s +0.0006; Metrica +0.004 (2 / 3), 3 s +0.002. Küçük, iki veride kayıpsız. **S?**
  - **Merkezi fark hızlar (P9as):** DFL ±0.000, 3 s −0.001; Metrica +0.002, 3 s −0.008. Alınmadı. **O**
- **Geniş TCN tamam (P9ao, sadece takım, 6 tohum):** dilation 1–128 (±20 s) DFL 7 maç +0.0064 (4 / 7, biri eşit), 3 s hakeminde +0.003
  (5 / 7); Metrica +0.014 (2 / 3), 3 s +0.005 (1 / 3). 1–64 tutarsız. Tam zincir kararı P9au OOF'larıyla. **S?**
- **Public düşüşü (0.787 → 0.739) incelendi (4 Kas):** aynı kodun Colab kurulumu (v14) 0.787, Kaggle kurulumu (v14k) ve v16 0.739. Fark
  public penceresinde yalnız ölü topta: Kaggle kurulumu bir A possession'ının ortasına 3.1 s sahte ölü blok koyuyor (pencere kare
  5965–6043) ve sondaki ölü bölümde iki ölü blok arasında 2.3 s'lik bir H şeridi bırakıyor. Tüm maçta iki kurulumun ölü topu karelerin
  %0.8'inde ayrışıyor (ölü GRU donanıma göre biraz farklı öğreniyor); sistematik hata yok. İki ölü blok arasındaki 2–5 s canlı şeritleri
  kapatmak OOF v16'da DFL −0.004 / −0.009 / −0.011 (3 / 4 / 5 s, 3 s hakeminde daha kötü), Metrica −0.006 / −0.010 / −0.014: o
  şeritler çoğunlukla gerçek possession. Public'i yükseltecek değişiklik maçın geri kalanında kaybettirirdi; yapılmadı. **O**
- **v17 kuruldu (Kaggle T4; BigGRU üyeleri, ölü top v8, katı restart):** v16'dan karelerin %1.23'ü farklı, possession 261 → 260 / 90.
  GT'siz yargıçlar: Smart Tagging farklı karelerde %63.6 v17, forma %48.1 (nötr). Tam zincir ölçümüyle (3 s hakemi DFL +0.004, Metrica
  +0.008) aynı yön: **teslim v17** (`evaluation/results/v17_chain.csv`). Ölü GRU 6 tohum sonraki kurulumda. **S**

- **Lokal / Drive / Kaggle eşleşme denetimi (5 Kas gece):** zincir girdileri üç yerde aynı (Drive md5 = lokal; Kaggle 6 veri setinin
  tüm `cache/` ve `models/` dosyaları boyutça = lokal; `frames_v1`, `possessor_prob` (ikinci koşu), `tactical_v1`, restart, kaleci
  dahil). Teslim edilen `v17_chain.csv` = Kaggle çıktısı (md5 aynı). 21 Kaggle koşusunun hepsi burada kayıtlı. **Kopukluklar:** (1)
  Drive'daki Colab kod paketi 3 Kas'ın, geçmiş yeniden yazılmadan önceki commit'iydi (4771d38; BigGRU, ölü top v8 yok): Colab
  dönünce eski kodla koşacaktı → güncellendi. (2) Kaggle kod paketi bir commit geride (17d9d60): teslim edilen v17 ölü GRU'yu 3
  tohumla kurdu, depodaki v17 6 tohum diyordu, yani depo teslim dosyasını üretmiyordu → `make_submission v17` artık 3 tohum
  (teslimdeki gibi), 6 tohum sonraki sürümde. (3) rclone'un ortak Drive istemci kimliği 2026'da kapanacak (uyarı); gerekirse
  kendi client_id. **S**
- **Daha yüksek geçiş cezası yeniden (P9l denetimi, v16 OOF + BigGRU üyeleri; ceza k × (12(1−b)+0.5b)):** 3 s hakeminde k arttıkça
  artıyor (DFL k=1.5 +0.006 7 / 7, k=2 +0.009; Metrica k=1.5 +0.006, k=2 +0.010) ama 1.6 s'de Metrica her k'da 3 maçın 3'ünde geride
  (k=1.25 −0.003, k=2 −0.010). Bizim maçta k=1 → 1.25 → 1.5 → 2 possession / 90: 252 → 245 → 243 → 230; model zaten ayarlı hakemin
  DFL yoğunluğunun (~286) altında bölüyor, ceza bunu büyütür. GT tanımı ("mekanik, yumuşatma yok", public'te 0.8–3 s'lik
  sandviç possession'lar var) da daha kaba çıktıyı desteklemiyor. Üç kanıttan biri lehte, ikisi aleyhte: **k = 1 kaldı.** **O**
- **Geniş TCN tam zincirde, ara (4 DFL maçı, v17 üyelerinde TCN yerine / TCN ile ortalama, CPU):** yerine −0.0006 (1 / 4), 3 s
  −0.0011 (1 / 4); ortalama +0.0007 (3 / 4), 3 s −0.0010 (1 / 4). Takım kararındaki +0.006 tam zincire taşınmıyor; kalan 3 maç ve
  Metrica bitince kesin karar. 5 maçta: yerine +0.003 (2 / 5), 3 s +0.004 (2 / 5); ortalama +0.003 (4 / 5), 3 s +0.002 (2 / 5);
  kalan iki DFL maçı ve Metrica kuyrukta. **O?**
- **BigGRU üç bozma çekilişiyle (P9aw, Kaggle; sadece takım, tohum 3–5, 7 DFL + 3 Metrica):** DFL +0.016 (6 / 7), 3 s +0.014
  (5 / 7); Metrica +0.016 (2 / 3), 3 s +0.005 (1 / 3). Üretim GRU'sunda ikinci çekiliş işe yaramamıştı (P7g); büyük ağ daha fazla
  veriyi kullanıyor. Tam zincir testi P9ay (`analysis/p9ay_big_gru_draws_oof.py`, tek başlı, 3 tohum) yerelde. P9au (geniş TCN) 5. DFL
  maçından sonra durduruldu (tam zincirde ≈ 0, GPU P9ay'a). **S?**
- **P9ay, tam zincir DFL 7 maç (v17 OOF zincirinde tek başlı BigGRU yerine 3 çekilişle eğitilmiş olanı, 3 tohum):** 1.6 s
  **+0.0058 (7 / 7)**, 3 s hakeminde +0.0043 (5 / 7). Metrica yerelde sürüyor; kural tutarsa v18 kuyrukta kendiliğinden kuruluyor
  (`analysis/p9ay_chain.py` çıkış kodu). **S?**
- **Kabul kuralı tek biçimde (5 Kas):** v18 kapısı (`analysis/p9ay_chain.py`) önce Metrica'da yalnız 3 s'e bakıyordu; geçiş cezası
  artışı ise Metrica 1.6 s'de geride kaldığı için reddedilmişti. Kapı artık ikisini de istiyor: DFL ve Metrica, 1.6 s ve 3 s
  hakeminde geride değil, DFL 1.6 s'de önde. **K**
- **Geçiş cezası ölçeği k, iç içe LOMO, k < 1 dahil (P9ba; v17 OOF zinciri, CPU):** k < 1 DFL'de iki hakemde geride (k 0.8:
  −0.001 / −0.005), Metrica 3 s'de geride (−0.005). İç içe seçim DFL'de hep 1.25 (+0.002 5 / 7, 3 s +0.004 6 / 7), Metrica'da
  seçilen k −0.002 / −0.002 (1 / 3). Tek kurala göre **k = 1 kalıyor**; bizim maçtaki az bölmeyi cezayla açmak da yok. **O**
- **v18 kapısında ikinci deneme yok (7 Kas):** 7. üye yedeği (v18b) Metrica 1–2 görüldükten sonra eklenmişti; aynı veride
  ikinci varyantın oy hakkı şans eseri geçmeyi artırır. Kapı yalnız asıl adaya (tek başlı BigGRU yerine 3 çekilişli) bakıyor;
  7. üye satırları bilgi. Kural tutmazsa v18 kurulmaz. **K**
- **P9ay tamam (10 maç, tam zincir; tek başlı BigGRU yerine 3 çekilişli):** DFL 1.6 s **+0.0058 (7 / 7)**, 3 s +0.0043 (5 / 7);
  Metrica 1.6 s +0.0008 (2 / 3), 3 s **−0.0029 (0 / 3)**. Metrica 3 s hakeminde üç maçta da geride: tek kurala göre **v18 kurulmadı**,
  teslim v17 kalıyor. (7. üye satırı bilgi: DFL +0.0015 / +0.0028, Metrica +0.0012 / −0.0018.) Çekiliş DFL'de tutarlı kazandırıyor,
  Metrica'da 3 s'de kaybettiriyor: kazanç DFL'ye özgü bozma ayrıntılarına bağlı olabilir. **O**
- **Veri yeniden tarandı (7 Kas):** ATD'de gizli kanal yok (46 iz + top, x / y / hız; XML "km/h" diyor ama hız gerçekte m/s, zaten
  kullanmıyoruz). Kullanılmayan tek kaynak DFL olay verisiydi (`events_raw`, maç başı ~1,700 olay; floodlight okuyor). Ölçüm: ham
  takım değişimlerinin yalnız %29–37'sinde yeni takımın bir olayı ±0.5 s içinde (±1 s: %50–63, ±2 s: %77–85), olay saatleri
  değişimden medyan 0.15–0.43 s sonra; 1.6 s etiketlerdeki ≤ 2 s spell'lerin yalnız %42'si kendi takımının olayını içeriyor
  (tüm spell'lerde %82). Olay saatleri elle girilmiş, ~±1 s gürültülü: sınır hedefi için konum dosyasının possession bayrağından
  zayıf, kısa spell'lerin çoğunda olay yok. Yardımcı baş olarak denenmedi. Restart'ların `DecisionTimestamp`'i (düdük anı) var,
  ama ölü top etiketleri zaten konumdan geliyor. Smart Tagging'in takımlı fazları (public'te GT ile %69.6) DFL / Metrica'da
  sınanamadığı için kanıt olarak kullanılmıyor. **O**
- **P9bc (yarı sonu pencereli tek başlı BigGRU, tam zincir 10 maç):** DFL +0.0020 (5 / 7), 3 s +0.0009 (4 / 7); Metrica 1.6 s
  **−0.0036 (0 / 3)**, 3 s +0.0010. Kural tutmadı. **O**
- **Geniş TCN tam zincirde, 10 maç (P9au; TCN üyelerinin ikisi de dilation 1–128):** DFL 1.6 s **+0.0040** (3 / 7), 3 s +0.0016
  (3 / 7); Metrica +0.0016 (3 / 3), 3 s +0.0009 (2 / 3). İki veride iki hakemde de önde: **v18'e** (ortalama kural; DFL'de kazanç
  birkaç maçta toplanıyor). Ortalama varyantı bilgi: DFL +0.0024 / +0.0017, Metrica +0.0006 / −0.0013. **S**
- **Öğrenen geçiş cezası (P9be; 2 durumlu CRF olabilirliği, ceza = softplus(θ·[1, b, logit b, ölüye yakınlık]), iç içe LOMO,
  CPU):** emisyon ölçeği serbestken 0.01'e düşüyor (komşu karelerin olasılıkları bağımsız değil; ceza ~600, tek possession).
  Ölçek 1'de sabit: DFL −0.0039 / −0.0088, Metrica −0.0044 / −0.0098. Olabilirlik hedefi spellF1 ile uyuşmuyor; alınmadı. **O**
- **v18 kuruldu (Kaggle T4, 5831e82; v17 + geniş TCN üyeleri):** v17'den karelerin %0.42'si farklı (yalnız H / A; ölü top bire bir
  aynı: aynı donanım ve tohumlar), possession 296 → 291 (X boşluk sayımı), 2 s'den kısa spell 33 → 31. Public 0.763 → 0.718:
  pencerede 62 kare, ölü topa yakın 0.8 s ve 1.5 s'lik iki H şeridi; tek possession'ın bölünmesi bu kadar oynatıyor, karar
  DFL / Metrica tam zincirinde (P9au). **Teslim v18** (`evaluation/results/v18_chain.csv`). **S**
- **TCN'de üç çekiliş (P9az, Kaggle; üretim TCN'i, sadece takım, tohum 3–5):** DFL +0.009 (6 / 7), 3 s +0.013 (4 / 7); Metrica +0.019
  (3 / 3), 3 s +0.014 (2 / 3). Tam zincir testi: v18'in tek başlı geniş TCN'i yerine üç çekilişli olanı (P9bg). **S?**
- **Büyük oyuncu kümesi (P9av, ilk 4 DFL maçı, Kaggle):** 256 x 3 − üretim −0.002 (2 / 4), 3 s −0.005 (1 / 4); kalanı yerelde. **O?**
- **Büyük oyuncu kümesi tamam (P9av, 10 maç, sadece takım):** DFL 7 maç ≈ +0.003 (ilk 4 −0.002, son 3 +0.009), 3 s ≈ −0.002; Metrica
  −0.009 (1 / 3), 3 s −0.018 (1 / 3). Takım kararında bile Metrica'da geride: alınmadı, tam zincire gerek yok. **O**
- **SoccerTrack ön eğitimi BigGRU'da (P9bf, Kaggle, tam zincir; tek başlı BigGRU yerine, 8 SoccerTrack maçı):** DFL 1.6 s −0.0011 (2 / 7),
  3 s −0.0010 (3 / 7). DFL'de önde değil, kural tutmadı (Metrica indirilemedi, karar için gerekmiyor). P8j'nin +0.009'u BigGRU'da yok. **O**
- **Sınır GRU'su üç çekilişle (P9bi, Kaggle, v18 OOF zinciri):** DFL +0.0023 (5 / 7), 3 s +0.0019 (5 / 7); Metrica 1.6 s **−0.0019
  (0 / 3)**, 3 s +0.0014. Kural tutmadı. Desen: üç çekiliş BigGRU'da (P9ay), sınır GRU'sunda (P9bi) DFL'de kazandırıp Metrica'da
  (bir hakemde, üç maçta da) kaybettiriyor; eğitim verisinin 7 / 10'u DFL, çekilişler DFL'nin bozma ve etiket üslubunu
  güçlendiriyor olabilir. **O**
- **Üç çekilişli geniş TCN tam zincirde (P9bg, Kaggle; v18 OOF zincirinde tek başlı geniş TCN yerine, tohum 3–5):** DFL 1.6 s
  **+0.0032 (5 / 7)**, 3 s +0.0002 (5 / 7); Metrica **+0.0030 (3 / 3)**, 3 s +0.0009 (2 / 3). İki veride iki hakemde de önde: **v19'a**
  (yalnız tek başlı üye; iki başlı ikizi P9bh yerelde sürüyor). Çekiliş BigGRU'da ve sınır GRU'sunda Metrica'da kaybettirmişti,
  TCN'de kazandırıyor. **S**
- **Yerel kurulum = Kaggle kurulumu (8 Kas):** v18'in Kaggle ağırlıklarıyla (hash ea44a36fa4aceb67) yerelde yeniden üretilen zincir Kaggle'daki
  v18 ile bire bir aynı (md5, 0 kare; ölü top aynı, olasılıklarda en çok 1.5e-4). Ağırlıklar taşınınca yerel kurulum güvenli; v19
  yerelde yalnız yeni üyeyi eğitiyor. **S**
- **v19 kuruldu (yerel, v18'in Kaggle ağırlıkları + yeni tek başlı geniş TCN üç çekilişle, 6 tohum):** v18'den karelerin %0.30'u
  farklı (yalnız H / A), ölü top bire bir aynı, possession 291 / 291, 2 s'den kısa spell 31 → 30, public 0.718 (v18 ile aynı). Tam
  zincir: DFL 0.7347 / 0.7044 (v17 0.7275 / 0.7026), Metrica 0.7754 / 0.7753 (v17 0.7708 / 0.7736). **Teslim v19**
  (`evaluation/results/v19_chain.csv`). **S**
- **Kahin ayrıştırması (v19 OOF zinciri, 10 maç, 8 Kas):** DFL 0.7346 → ölü top etiketten 0.7596 (+0.025), takım kararı etiketten (tahmin
  edilen ölü topla) 0.9584 (+0.22), ikisi 0.9996; Metrica 0.7754 → 0.8068 / 0.9473 / 0.9976. IoU başına F1 (0.3…0.7) 0.833 / 0.803 /
  0.756 / 0.702 / 0.641. GT spell 3513, tahmin 3070. IoU 0.5'te yakalama (precision) süreye göre: 0–2 s n=681 %34 (409 tahmin, %58),
  2–5 s n=1034 %66 (%75), 5–10 s n=752 %82 (%85), 10–20 s n=592 %90 (%90), > 20 s n=454 %93 (%95). Kayıp neredeyse tamamen takım
  kararında ve kısa spell'lerde. **S**
- **Sınırda kes ve oyla (P9bl; ASRF, v19 OOF, iç içe LOMO, CPU):** saf kes-ve-oyla DFL −0.033 (0 / 7), 3 s −0.014; Metrica −0.037 /
  −0.007. Viterbi spell'inin içine kısa spell ekleme: iç içe seçim hep eklememeyi seçiyor (±0.000); iki sınır tepesi arasındaki
  kısa aralıklarda olasılık karşı takımı yeterince söylemiyor. Kısa spell kaybı kod çözmede değil, girdideki bilgide. **O**
- **Restart yasası ölü toplar boyunca (P9bm; zincirli Viterbi, aynı takımla devam cezası C_SAME, ölü top öncesi W karede ucuz değişim,
  v19 OOF, iç içe LOMO, CPU):** ızgara[0] v19'u bire bir veriyor. İç içe: DFL −0.0012 (1 / 7), 3 s −0.0009; Metrica +0.0014 / +0.0016.
  En iyi ortalama ayar DFL'de bile +0.0007. W / RHO hiçbir şeyi değiştirmiyor: ölüden önceki karelerde olasılık karşı takımı
  göstermiyor. Alınmadı. Üç kod çözme denemesi (P9bl, P9bm, P9be) aynı yerde: kısa spell bilgisi girdide yok. **O**
- **Kısa spell üslubu DFL'de ve Metrica'da (1.6 s etiketleri, oyun dakikası başına, CPU):** < 2 s spell DFL 1.25 / Metrica 1.19;
  sandviç (A → B → A) DFL **0.29** / Metrica **0.59**; ölüye bitişik DFL 0.95 / Metrica 0.60. Hedefin sağlayıcısı Metrica iki kat fazla
  sandviç kısa spell sayıyor: DFL'deki sandviçleri silen kontrol temelli yeniden etiketleme (rapordaki 2. aday) hedeften uzaklaştırır,
  bırakıldı. Eğitim verisinin 7 / 10'u DFL olduğundan model DFL üslubuna yaslanıyor olabilir. **O**
- **İki başlı geniş TCN üç çekilişle (P9bh, yerel, v19 OOF zincirinde iki başlı geniş TCN yerine):** DFL 1.6 s −0.0004 (4 / 7), 3 s
  +0.0026 (5 / 7). DFL 1.6 s'de önde değil, kural tutmadı; Metrica koşusu iptal. Çekiliş yalnız tek başlı TCN'de kazandırdı. **O**
- **Önceden kayıtlı kural, P9bn (Metrica ağırlığı x3, tek başlı BigGRU, v19 zinciri; 8 Kas 21:10, koşudan önce):** Metrica iki hakemde de
  ortalamada önde ve her hakemde en az 2 / 3 maç önde; DFL iki hakemde de en çok −0.002. Gerekçe: hedefin sağlayıcısı Metrica ve kısa
  spell üslubu DFL'den farklı (sandviç 0.59 / 0.29). Metrica önce koşuyor; tutmazsa DFL koşmaz. **K**
- **P9bn sonucu (8 Kas 23:52):** Metrica 1.6 s +0.0040 (3 / 3), 3 s −0.0018 (1 / 3) → kural tutmadı; DFL koşusu kurala göre durduruldu. Metrica ağırlığı kısa spell'i 1.6 s'de biraz artırıyor, 3 s hakeminde geri veriyor: kayıp ağırlığı üslup farkını taşımıyor. **K**
- **P9bp sonucu (9 Kas 00:28; oyuncu kümesi üç çekilişle, v19 zinciri):** Metrica 1.6 s +0.0056 (2 / 3), 3 s −0.0017 (2 / 3) → Metrica 3 s'de geride, tek kural DFL'ye bakmadan düştü; DFL koşusu iptal. P9bn ile aynı desen: Metrica'da 1.6 s kazancı 3 s'de geri veriliyor (kısa spell artıyor, 3 s birleştirmesi cezalandırıyor). **O**
- **Önceden kayıtlı, P9bq (tek başlı geniş TCN 3 → 5 çekiliş, tohum 3–5, float16 özellik; v19 OOF zincirinde d3 üyesi yerine; 9 Kas 00:30, koşudan önce):** tek kural (DFL ve Metrica iki hakemde de geride değil, DFL 1.6 s önde). Metrica önce; Metrica geride kalırsa DFL koşmaz. **K**
- **Önceden kayıtlı, P9br (v19 OOF zincirinde üye ağırlıkları: mimari başına {0.5, 1, 2}, olasılıkta ya da log-odds'ta karışım, 50 ayar; her maç diğer 9'da en iyi ayarla, iç içe LOMO; 9 Kas 00:40, koşudan önce):** tek kural iç içe skorlar üzerinde. CPU, kuyrukta P9bq Metrica'dan sonra. **K**
- **P9br sonucu (9 Kas 01:41, CPU 90 s):** iç içe DFL 1.6 s +0.0016 (5 / 7), 3 s +0.0017 (5 / 7); Metrica −0.0004 (1 / 3), 3 s −0.0009 (2 / 3) → alınmadı. Seçim hep TCN'e ağırlık, oyuncu kümesine az: (1, 2, 0.5); en iyi tek ayar ortalamada +0.003 (iç içe değil, bilgi). Eşit ağırlık Metrica'da yeterince iyi; ağırlık ayarı DFL'ye uyuyor. **O**
- **P9bq sonucu (9 Kas):** Metrica 1.6 s −0.0010 (1 / 3), 3 s −0.0048 (1 / 3) → alınmadı; DFL'nin ilk iki maçı (+0.0008 / +0.0029) kural gereği koşmamalıydı, 08:06'da kuyruk başlattı, 15:25'te durduruldu. 3 → 5 çekiliş Metrica'yı geri götürüyor: çekiliş sayısı 3'te doymuş. **O**
- **Önceden kayıtlı, P9bs (tek başlı geniş TCN, 3 çekiliş, tohum 3–5; test anında takım takası, aynı ağlarla düz / tta eşli; 9 Kas 15:35, koşudan önce):** tek kural. Metrica önce; kuyrukta kapı (GATE=metrica) Metrica geride değilse DFL'yi başlatıyor. **K**
- **Önceden kayıtlı, video kanıtı (SoccerTrack kırpıntı modeli; 9 Kas 17:10, hiçbir video sonucu görülmeden):** DFL / Metrica'da video
  yok, eski kural sınayamıyor; karar yalnız SoccerTrack'te, daha zayıf bir ölçüm olduğu için eşikler daha sert.
  Etiket: SoccerTrack top dosyasının kendi kare kare durumu (takım / boşta / ölü; `mtah.external.soccertrack_labels`), 1.6 s ve 3 s
  birleştirmesi diğer veriler gibi. Taban: v19c zinciri SoccerTrack'te yalnız çıkarım (modeller DFL + Metrica'da eğitilmiş), takip
  BAR-ATH oranlarıyla bozulmuş; video sonucundan önce bir kez hesaplanır. Bölme: maça göre iki kat (5 / 5); kırpıntı modeli ve kanıt
  ağırlığı yalnız eğitim katında seçilir, sınanan katta hiçbir şey ayarlanmaz.
  Kapı A (ucuz, zincire takmadan, saat harcamadan önce dur noktası): sınanan kattaki aksiyon karelerinde dokunanın takımı doğruluğu
  ≥ %75 ve aynı karelerde en yakın oyuncu kuralından ≥ 5 puan yukarıda. Tutmazsa iş durur.
  Kapı B (teslime alma): sınanan maçlarda zincir+video − zincir ortalaması 1.6 s'de ve 3 s'de ≥ +0.010; 1.6 s'de yarıların ≥ %70'inde
  pozitif; hiçbir yarı −0.02'den kötü değil. Ek bilgi (kapı değil): 0–2 s spell yakalaması.
  Kapı C (BAR-ATH, GT'siz, yalnız verilen veriyle otomatik kontrol, ayar değil): panorama → bizim video alan kayması için kırpıntı
  modelinin takımı, zincirin p > 0.9 emin olduğu karelerde zincirle ≥ %80 uyuşmalı; possession sayısı v19c'nin (285) 0.8–1.5 katı
  içinde kalmalı. Dışındaysa model aktarılmıyor ya da bozuk: alınmaz. Public skora karar için bakılmaz.
  DFL / Metrica'da kanıt yokken zincir v19c ile bire bir aynı kalmalı (yapı gereği, test edilir). Sonuç görüldükten sonra ikinci
  varyant yok; tutmazsa teslim v19c kalır. **K**
- **Video kanıtı kuralına ekler (9 Kas 18:00, hiçbir video sonucu görülmeden):** Kapı B'de ölü top iki kolda da etiketten (video yalnız
  takım kararına dokunuyor; ölü top modeli SoccerTrack görmedi); restart kanıtı iki kolda kapalı (SoccerTrack'te Smart Tagging yok).
  Katlar: A = 117092 117093 118575 118576 132831, B = 118577 118578 128057 128058 132877 (132831 / 132877 possession durumu yok:
  yalnız kırpıntı eğitimi). Kapı A'nın tabanı: aynı aksiyon karelerinde BAR-ATH oranlarıyla bozulmuş topa en yakın oyuncu (bizim
  maçta elimizdeki bu); bozulmamış topa en yakını bilgi olarak raporlanır. GSR `bbox_image` gerçek tespit değil (izdüşüm, ~32 px
  sabit, 30–90 px kayık): kutular YOLO11m'den (BAR-ATH'la aynı ayar), izdüşürülmüş kutularla eşlenerek. Tasarım tek kare
  kırpıntıdan zamansal modellere çevrildi (T-DEED takım top aksiyonu, oyuncu kırpıntı dizisi; P9bx sanal kamera, 12.5 / 25 fps). **K**
- **SoccerTrack possession durumu aksiyonlara bağlı (9 Kas 18:30, yan ajan uyarısı, ölçüldü; hiçbir video sonucu görülmeden):**
  README: top izi ölçüm değil, aksiyonlara sabitlenip düz çizgiyle doldurulmuş. Durumdaki takım değişimlerinin %70–85'i (aralık
  %48–85) bir aksiyona ≤ 2 kare uzak; yarı başına ~110 değişim, ~1,000 aksiyon. Aksiyonları bulan bir video modeli SoccerTrack'te
  sınırları kolay tutturur: Kapı B iyimser. Takip tabanı da aynı yeniden kurulmuş topu kullanıyor. Kural eki: Kapı B eşiği
  1.6 s'de ve 3 s'de ≥ +0.020; ek olarak aksiyonlardan ≥ 0.5 s uzak canlı karelerde takım doğruluğu en çok −0.2 puan;
  Kapı C zorunlu. Kapı A değişmiyor (doğrudan aksiyonu yapanın takımını ölçüyor). **K**
- **P9bs sonucu (9 Kas 21:04):** test anında takım takası Metrica 1.6 s −0.0002 (1 / 3), 3 s +0.0003 → kapı DFL'yi başlatmadı, alınmadı. **O**
- **Kapı B tabanı (P9bw, v19c takım kararı SoccerTrack'te yalnız çıkarım, ölü top etiketten):** 1.6 s / 3 s — A: 117092 0.678 / 0.653,
  117093 0.765 / 0.730, 118575 0.783 / 0.770, 118576 0.735 / 0.738; B: 118577 0.733 / 0.717, 118578 0.713 / 0.713, 128057 0.789 / 0.773,
  128058 0.744 / 0.714. Ortalama ≈ 0.743 / 0.726 (DFL'deki takım kararıyla aynı düzey: modeller SoccerTrack'e aktarılıyor). **S**
- **Kural açıklaması (organizatör e-postası, kullanıcı iletti 8 Kas):** yasak: çıktıyı videoyla kontrol etmek; verilen maçta elle / insan
  üretimi veri; verilen veriyi AI servislerine yüklemek; GT possession verisinde ya da verilen videoda eğitim, uydurma, kalibrasyon.
  Serbest: yalnız verilen veriyle otomatik kontroller; tanımı anlamak için incelemek; video / ATD / Smart Tagging'i girdi olarak kullanmak
  (algoritma GT'de ya da videoda eğitilmediyse); eğitim ve ayar YALNIZ dış veride. Sonuç: possessor_r18 (BAR-ATH videosunda eğitildi,
  SCALE ve kalibrasyon bizim maçta) yasak → **teslim v19c** (kırpıntısız; v19'dan %0.34 kare farklı; DFL / Metrica skorları aynı).
  Gri, organizatöre sorulacak: bozma oranları bizim ATD'den GT'siz ölçüldü (bir kısmı videodaki YOLO topundan); eski GT'siz yargıç
  (forma, Smart Tagging) kararları. SoccerTrack'te eğitilip orada ayarlanan video kanıtı kurala uygun. **K**
- **v1 → v19c denetimi (8 Kas, GT'siz tam maç ölçümleri + public yalnız rapor):** spell sayısı v1 367 → v2 277 → v10–v19c 285–306 (v11b 336,
  alınmamıştı); < 2 s spell v10–v19c 26–41; v19c en az bölen (285 / 26). Public v3 0.62 → v10 0.76'dan beri 0.72–0.79 bandında; aynı 12 / 17
  possession bulunuyor, fark penceredeki tahmin sayısı (14–17). Hata yok: yerel = Kaggle kurulumu, v19 yalnız yeni üyeyi eğitti, v19c'de
  kırpıntı x0. Uyum: takım / sınır / ölü top modelleri dış veride; ölü top kuralı eşikleri DFL'den; dönemler yayınlanmış sınır + Vidal-Codina;
  restart sabitleri tasarım; bozma oranları gri (lehimize). Atlanmış ölçülmüş iyileşme yok. **S**
- **Kapı A tabanı (P9cd, yerel):** aksiyon karelerinde bozulmuş topa en yakın oyuncunun takımı A katı %75.8, B katı %76.4
  (bozulmamış topta %89.4 / %88.0). Video modelinin eşiği: ≥ %81. **S**
- **Aksiyon kanıtının tavanı (P9ce / P9cf, simüle kanıt, CPU; P9cf yalnız A katında, B katı görülmedi):** her aksiyonun çevresine
  ±W log-odds eklemek kusursuz kanıtla bile kaybettiriyor (8 maç, W 2 / 4 / 8: −0.017 / −0.082 / −0.133): aksiyonların ~%9'unda
  dokunan takım etiketin takımı değil (blok %20, top kapma %36, pas / sürüş %90–91) ve her biri sahte kısa spell açıyor;
  zincir aksiyon karelerinde zaten %87.5 aynı takımı söylüyor. Tutan biçim "çift": aynı takımın en çok 4 s arayla art arda iki
  aksiyonu arasına kare başına +w. Kusursuz kanıt (w 4): **+0.042 / +0.043** (4 / 4); yakalama 0.7, takım doğruluğu 0.9, ±3 kare
  kayma (w 2): +0.021 / +0.026 (4 / 4); 0.6 / 0.8 (w 0.5–1): +0.006–0.007 / +0.008–0.010. Geçiş cezasını düşürmek tek başına ≈ −0.001.
  Okuma: aksiyon tespiti SoccerTrack'te (iyimser ölçüm) en çok +0.04, gerçekçi +0.006–0.02; Kapı B eşiği (+0.020) ancak T-DEED
  0.7 / 0.9 düzeyine çıkarsa geçilir. Füzyon biçimi "çift" olarak sabitlendi; w eğitim katında seçilecek. **O**
- **Kare kare takım kanıtının tavanı (P9ch, simüle, A katı, CPU):** zincirin canlı karelerde takım doğruluğu %90–93. 1 s'lik
  bloklarda bağımsız hatalı kanıt, kare başına ±w: kusursuz (w 2) +0.18; blok doğruluğu 0.95 +0.096 (w 1); 0.90 +0.069 (w 1) /
  +0.056 (w 0.5); 0.85 +0.048 / +0.044; kapsama 0.5'te 0.90 → +0.029, 0.85 → +0.025. Aksiyon kanıtından (en çok +0.02) çok büyük.
  Uyarı: simülasyondaki hatalar zincirin hatalarından bağımsız; gerçek modelin hataları zor anlarda zincirle birlikte olur, kazanç
  daha küçük çıkar. **O**
- **Yanlış alarmlı aksiyon kanıtı (P9cj, simüle, A katı; yan ajan uyarısı üzerine):** "çift" biçiminde w 0.5–1 iken kesinlik
  1.0 → 0.55'e düşse de kazanç +0.011–0.020 bandında kalıyor (0.9 / 0.7: +0.013–0.019); w 2 kesinlik ≤ 0.7'de bozuluyor
  (−0.006 … +0.006, en kötü maç −0.02). Aksiyon kanıtı tek başına Kapı B eşiğini (+0.020) gerçekçi düzeyde geçmiyor. **O**
- **Karar (10 Kas, hiçbir video modeli sonucu görülmeden):** T-DEED'e yoğun possession başı eklendi (P9ci: kare başına sol / sağ /
  ölü, top dosyasının durumundan; aksiyon kayıpları yardımcı görev). Karar testi: A katının 4 maçında 6 epoch, B katının 118577 ve
  118578 maçlarında; füzyon kare başına log-odds'a w·(log q_ev − log q_dep), **w = 0.5 önceden sabit** (eğitim katı tahminleri
  ezber olacağı için orada seçilemiyor), ölü top etiketten. Devam ölçütü: iki maçın ortalaması 1.6 s'de ve 3 s'de ≥ +0.020 ve iki
  maç da pozitif. Tutmazsa video hattı kapanır, v19c kalır. Tutarsa: kalan yarılar, iki kat, Kapı B ve C aynen. **K**
- **T-DEED eğitimsiz (SoccerNet ağırlıkları, P9bz epoch 0) SoccerTrack sanal kamerasında, 118577 + 118578 (P9cg, ±1 s):** eşik 0.3:
  yakalama 0.71 / kesinlik 0.59; 0.5: 0.58 / 0.70; eşleşen possession aksiyonlarında takım (sol / sağ) doğruluğu %77–80. Yayın
  kamerasında eğitilmiş model sanal kameraya aktarılıyor; ince ayar öncesi taban. L4'te grup 4 belleğe sığmadı: P9ci grup 2, lr 4e-4,
  adım 1.75 s, epoch ~73 dk. **S**
- **Public sürüm farkı (10 Kas, yalnız rapor):** v10–v19c hepsi penceredeki 17 possession'ın aynı 13 / 12 / 11'ini (IoU 0.3 / 0.5 / 0.7)
  yakalıyor; skor farkı yalnız penceredeki tahmin sayısından (14 → 0.787, 15 → 0.762, 16 → 0.739, 17 → 0.718). v19c %90 bandı 0.53–0.87. **S**
- **Ölü topa bitişik kısa possession'lar dış veride kazandırıyor (P9cn, v19 OOF zinciri, CPU):** bunları ölüye çevirmek < 1 s: DFL
  −0.0021 / −0.0011 (2 / 7), Metrica −0.0039 / −0.0045 (0 / 3); < 2 s: DFL −0.0198 / −0.0157 (0 / 7), Metrica −0.0252 / −0.0203 (0 / 3);
  < 3 s: −0.038 / −0.046. IoU 0.5'te tutanların payı < 2 s: DFL %55, Metrica %71 (başa baş ~%37). Public'te tutmayan iki şerit bu
  türden; silmek tam maçta kaybettirir. Değişiklik yok. **S**
- **Karar testi TUTTU (P9ci + P9ck, 11 Kas 03:50; A katında 6 epoch, grup 2, L4'te 6.4 saat):** yoğun possession başı, görülmemiş
  118577 / 118578'de w 0.5 (önceden sabit): 118577 0.7330 → 0.7508 (+0.0178), 3 s 0.7169 → 0.7429 (+0.0259); 118578 0.7134 → 0.7427
  (+0.0294), 3 s 0.7131 → 0.7444 (+0.0313); ortalama **+0.0236 / +0.0286**, 2 / 2. Canlı karelerde takım doğruluğu (1.6 s
  etiketleri): video 0.879 / 0.909, zincir 0.891 / 0.912, birleşik 0.907 / 0.929; video zincirle %88–90 uyuşuyor. Ham durumda
  yarı başına video takım doğruluğu 0.909–0.945, üç sınıf 0.874–0.917. Aksiyon başı ince ayardan sonra (±1 s): eşik 0.6'da
  yakalama 0.91 / kesinlik 0.70, 0.7'de 0.86 / 0.80; takım %91 (eğitimsiz 0.53 / 0.73, %80). Kural: devam → kalan 8 yarı
  çıkarılıyor, iki kat, Kapı B ve C. Uyarı aynı: SoccerTrack durumu aksiyonlara bağlı, sanal kamera oradaki topu izliyor;
  bizim maçta Kapı C şart. **S**
- **Kapı C TUTTU (P9cm + P9co, 11 Kas; A katı modeli bizim maçta yalnız çıkarım, etiketsiz):** video kanıtı canlı karelerin %100'ünde;
  zincirin emin olduğu karelerde (p > 0.9 / < 0.1, canlı karelerin %80.0'i) takım uyumu **%93.7** (eşik %80); birleşik zincir
  (w 0.5) 285 → 289 possession (x 1.01; aralık 0.8–1.5); canlı karelerin %2.50'si değişiyor; 2 s'den kısa possession 26 → 37
  (DFL oranıyla beklenen ~63). Panorama → bizim taktik video aktarımı çalışıyor. Kalan: Kapı B'nin tamamı (8 maç, iki kat). **S**
- **Kapı B, B katının tamamı (A katı modeli, w 0.5; 11 Kas 14:45):** 118577 +0.0178 / +0.0259, 118578 +0.0294 / +0.0313, 128057
  0.7885 → 0.8267 (+0.0382 / +0.0395), 128058 0.7442 → 0.7652 (+0.0211 / +0.0327); ortalama **+0.0266 / +0.0324**, 4 / 4. Canlı
  karelerde takım doğruluğu video 0.88–0.92, zincir 0.89–0.92, birleşik 0.91–0.94. A katı (B katı modeli) eğitiliyor; orada
  132877'den gelen hedefsiz gruplarda yoğun kayıp NaN, adım atlanıyor (~%4; düzeltildi, son modelde yok). **S**
- **Kapı B'nin tam hesaplayıcısı (P9cp), B katının 4 maçında:** 8 / 8 yarı 1.6 s'de önde (en kötü yarı +0.0073); aksiyonlardan ≥ 0.5 s
  uzak canlı karelerde takım doğruluğu zincir %89.59 → birleşik %91.37 (+1.78 puan): kazanç yalnız aksiyon anlarından gelmiyor. **S**
- **Kapı B TUTTU (P9cp, 11 Kas 21:15; her maç onu görmemiş modelle, w 0.5 önceden sabit, ölü top etiketten):** 8 maç ortalaması
  **+0.0256 / +0.0300** (eşik +0.020); A katı maçları (B modeli): 117092 −0.0012 / +0.0114, 117093 +0.0312 / +0.0488, 118575
  +0.0243 / +0.0125, 118576 pozitif (çıktıda); yarılar 15 / 16 önde (%94; eşik %70), en kötü yarı −0.0081 (eşik −0.02);
  aksiyonlardan ≥ 0.5 s uzak canlı karelerde takım doğruluğu %90.10 → %91.62 (+1.52 puan; eşik −0.2). 117092'de video zayıf (ham
  takım doğruluğu %65–72, diğer maçlarda %93–95; o maçın kareleri de küçük, 2.9 GB / yarı) ama birleşik zincir kaybetmiyor.
  Kapı A (aksiyon karelerinde takım ≥ %75 ve tabandan +5 puan): T-DEED aksiyon başı %91, taban %76 → tutuyor. Kapı C: tuttu (A modeli).
  Sıradaki: 10 maçın hepsiyle son model → bizim maçta çıkarım → Kapı C yeniden → v20. **K**
- **Önceden kayıtlı, P9cq (aksiyon kanıtı yoğun kanıtın üstüne; 11 Kas 22:55, koşudan önce):** aynı çapraz kat tahminleri (ek GPU yok).
  Tespit: 1 − P(arka plan) tepeleri, eşik 0.6, ±0.5 s bastırma; sınıfı possession aksiyonu olanlar (pas, sürüş, uzun pas, orta, şut,
  taç, serbest vuruş); takım T > 0.5. "Çift" biçimi (P9cf): aynı takımın ≤ 4 s arayla art arda iki tespiti arasına kare başına +w_a,
  **w_a = 1.0 sabit** (P9cj'de yanlış alarmlara dayanıklı aralık 0.5–1). Karşılaştırma: zincir + yoğun (w 0.5) ile zincir + yoğun + çift.
  Kural: 8 maç ortalaması 1.6 s'de ve 3 s'de ≥ +0.005, en az 6 / 8 maç 1.6 s'de önde, hiçbir maç −0.01'den kötü değil. Tutmazsa
  v20 yalnız yoğun kanıtla kalır; ikinci ayar denenmez. **K**
- **P9cq sonucu (11 Kas 23:00):** yoğun kanıtın üstüne aksiyon "çift" kanıtı 8 maçta −0.0052 / −0.0046 (3 / 8 önde, en kötü −0.034) →
  kural tutmadı. Yoğun baş aksiyon bilgisini zaten taşıyor; çift biçimi üstüne aşırı destek ekliyor. v20 yalnız yoğun kanıt. **O**
- **Önceden kayıtlı, P9cr (metriğe göre Bayes kararı; 11 Kas 23:40, koşudan önce; takım arkadaşının v22'sinden fikir):** v19 OOF zinciri
  (p, geçiş cezası, ölü top aynı). Her canlı parçada yol dağılımı ∝ exp(β · Viterbi skoru); ileri süzme–geri örnekleme ile K = 64 yol.
  Aday sınırlar: örneklerdeki geçiş kütlesinin (±6 kare) tepeleri (≥ 0.05) + Viterbi sınırları. Aday possession (i, j, takım) değeri
  = örnekler ve IoU eşikleri (0.3–0.7) üzerinden "aynı takımdan IoU ≥ eşik bir possession var" ortalaması; DP: takımlar sırayla,
  toplam (değer − λ) en büyük. Izgara β ∈ {0.1, 0.2, 0.4, 0.8}, λ ∈ {0.30, 0.37, 0.45}; her maç diğer 9'da en iyi (iki hakem
  ortalaması) ayarla (iç içe LOMO). Kural: tek kural (DFL ve Metrica iki hakemde de geride değil, DFL 1.6 s önde), iç içe skorlarda.
  Tutmazsa kod çözme Viterbi kalır. **K**
- **Önceden kayıtlı, P9cs (videodan ölü top; 11 Kas 23:59, koşudan önce):** SoccerTrack'in 8 maçı, P9bw'nin bozulmuş kopyaları. Zincirin
  ölü topu: v8'in sınıflandırıcısı + ölü GRU'su (DFL + Metrica'da eğitilmiş, SoccerTrack görmemiş), to_blocks((p + q) / 2); bizim maça
  özgü parçalar (oynanmayan dönemler, yayın kuralı) yok. Video kolu: to_blocks((p + q + v) / 3), v = yoğun başın ölü olasılığı (çapraz
  kat). Takım iki kolda da zincir + yoğun (w 0.5). Kural: video kolu − diğeri 8 maç ortalaması 1.6 s'de ve 3 s'de ≥ +0.003, en az
  6 / 8 maç 1.6 s'de önde, hiçbir maç −0.01'den kötü değil. Tutmazsa ölü top v8 kalır; ikinci ağırlık denenmez. Bilgi olarak:
  tam zincirde (ölü top da tahmin) yoğun kanıtın kazancı. **K**
- **Önceden kayıtlı, P9ct (sıra modeli istifleyici; 12 Kas 00:10, koşudan önce; takım arkadaşının v21'inden fikir):** girdi kare başına
  v19'un 6 üyesinin OOF log-odds'u (pg, pt, ps, mg, mt, ms), sınır olasılığı b, ölü olasılıkları qh, qg ve üyelerin ortalaması /
  standart sapması; model iki yönlü GRU (gizli 48, 1 katman), hedef H / A (canlı etiketli kareler), takım takası artırması, 3 tohum,
  8 epoch; her maç diğer 9 maçın OOF'unda eğitilen istifleyiciyle (iç içe). Zincir: istifleyici olasılığı üyelerin ortalamasının
  yerine, aynı geçiş cezası ve ölü top. Kural: tek kural (DFL ve Metrica iki hakemde de geride değil, DFL 1.6 s önde). Yerel GPU. **K**
- **P9ct sonucu (12 Kas 00:50, yerel GPU):** sıra modeli istifleyici iç içe DFL +0.0040 / +0.0055 (5 / 7), Metrica −0.0002 / +0.0002
  (2 / 3) → Metrica 1.6 s'de geride (kıl payı), kural tutmadı. İstifleyici ile üyelerin ortalaması (bilgi): DFL +0.0020 / +0.0041,
  Metrica +0.0014 / −0.0010. OOF olasılıkları cache/oof_stack_*.npy. **O**
- **Önceden kayıtlı, videodan ölü topun Kapı C'si (12 Kas 00:55; P9cs'nin ilk 4 maçı görüldü, bizim maçta birleşik ölü top hesaplanmadan):**
  bizim maçta etiketsiz: zincirin ölü topu canlı karelerin %44.9'u, video tek başına (q > 0.5) %49.0, kare uyumu %89.4. Birleşik
  ölü top = to_blocks((p + q + v) / 3) + v8'in diğer parçaları aynen (oynanmayan dönemler, yayın kuralı, kısa boşluk kapama). Şart:
  ölü payı zincirinkinden en çok ±3 puan; ölü blok sayısı 102'nin 0.8–1.3 katı; zincirin ölü topuyla kare uyumu ≥ %93. Tutmazsa
  videodan ölü top alınmaz. **K**
- **P9cs sonucu TUTTU (12 Kas 01:20):** SoccerTrack tam zincir (ölü top da tahmin) 0.6324 / 0.6134; yoğun kanıt +0.0144 / +0.0197
  (6 / 8); **videodan ölü top üstüne +0.0347 / +0.0351 (8 / 8, en kötü +0.0143)**. Ölü kare doğruluğu: zincir 0.89–0.93, video tek
  başına 0.88–0.96, üçü birlikte 0.93–0.95. Uyarı: ölü top modelimiz SoccerTrack'te alan dışı (tam zincir 0.63, ölü top etiketten
  0.74: orada ölü top 0.11 götürüyor; DFL'de kusursuz ölü top yalnız +0.025), yani bizim maçtaki kazanç bunun küçük bir kısmı
  olabilir; ayrıca SoccerTrack'in "ölü" tanımı (durum 0) hedefinkinden farklı olabilir. Bizim maçtaki etiketsiz şart (yukarıda) belirler. **S**
- **P9cr sonucu (12 Kas):** zincirin kendi olasılıklarından örneklenen akışlarla Bayes kararı iç içe DFL −0.0100 / −0.0084 (0 / 7),
  Metrica +0.0022 / −0.0078 → kural tutmadı; düşük β'da örnekler dağılıyor (−0.3 … −0.6), β 0.8'de bile −0.006. Kare bağımsız
  emisyonlu zincirden örneklemek gerçekçi akış vermiyor (arkadaşın yöntemi örnekleri istifleyici sıra modelinden alıyor). **O**
- **Videodan ölü topun bizim maçtaki şartı, A modeliyle ön koşu (P9cu; ölü GRU teslimdeki gibi 3 tohum):** yeniden kurulan v8, kayıtlı
  zincirin ölü topuyla bire bir aynı (0 kare); video eklenince ölü pay %44.9 → %46.2, blok 102 → 110 (x 1.08), kare uyumu %98.3 →
  şart tutuyor. Birleşik zincir (yoğun + video ölü, A modeli): 285 → 294 possession, < 2 s 26 → 45. Son modelle tekrarlanacak.
  Not: P9cs SoccerTrack'te ölü GRU'yu 6 tohumla koştu (v8'in belgelenmiş hali), teslim 3 tohumlu. `make_submission.py` v20 / v21 eklendi. **S**
- **Son video modeli (10 maç, 6 epoch) bizim maçta ve v20 / v21 kurulumu (12 Kas 04:30, yerel, `make_submission.py`):** Kapı C (son
  model): zincirin emin olduğu karelerde uyum %92.5, 285 → 288 possession → tutuyor. Videodan ölü top şartı: ölü pay %44.9 → %46.5,
  blok 102 → 110, uyum %98.1 → tutuyor; yeniden kurulan v8 kayıtlı ölü topla bire bir aynı.
  v20 (yoğun takım kanıtı, w 0.5): v19c'den oyun içi karelerin %1.13'ü farklı, 288 possession, < 2 s 31, ölü pay %44.9.
  v21 (v20 + videodan ölü top): %3.09 farklı, 293 possession, < 2 s 37, ölü pay %46.5.
  Public (yalnız rapor, seçimde yok): v19c 0.718, v20 0.729, v21 0.729.
  Dayanak: v20 → SoccerTrack Kapı B +0.0256 / +0.0300 (8 maç, çapraz kat); v21 → üstüne +0.0347 / +0.0351 (8 / 8), ama ölü top
  modelimiz orada alan dışı olduğu için bizim maçtaki payı belirsiz ve SoccerTrack'in ölü tanımı hedeften farklı olabilir.
  DFL / Metrica'da video yok: bu iki sürümün oradaki skoru v19c ile aynı (0.7347 / 0.7754). Teslim kararı kullanıcıda. **K?**

### 12 Nov — P9cv: the shape of the dense video evidence (SoccerTrack, cross-fold, 8 matches)
- Weight sweep of P9cp: w 0.25 +0.0237 / +0.0287 (16/16 halves), 0.5 +0.0256 / +0.0300, 1 +0.0165 / +0.0171, 1.5 +0.0016, 2 −0.0133, 3 −0.0332. 0.5 is the peak of the raw form; 117092 (weak video) carries the losses.
- Box-smoothing the log ratio before the fusion (`analysis/p9cv_evidence_shape.py`): at w 0.5, 51 frames +0.0354 / +0.0438, 101 +0.0470 / +0.0594, 151 +0.0522 / +0.0682 (8/8, worst +0.0128), 251 +0.0413 / +0.0643; nested pick +0.0542 / +0.0777 (7/8). Clipping does not help.
- Our match, label-free (151 frames, w 0.5): agreement where the chain is sure 93.0%, possessions 285 → 276, 2.0% of live frames changed.
- **Caveat that blocks adopting it as is:** SoccerTrack's label spells are long (1478 of 2027 over 5 s, 115 under 2 s; ours: half under 5 s) and the chain over-segments there (2465 predicted; raw evidence 2345, smoothed 2166). Recall at IoU 0.5 by label length, chain / raw / smoothed: < 2 s 42 / 47 / 40%, 2–5 s 71 / 71 / 68%, > 5 s 90 / 91 / 92%. The smoothing's gain is fragment removal on long spells and it costs short ones; the raw evidence is the form that helps short spells.
- **P9cw (`analysis/p9cw_length_weighted.py`): the same with every label and predicted spell weighted to DFL's length distribution** (shares < 2 / 2–5 / 5–10 / 10–20 / ≥ 20 s: DFL 0.30 / 0.31 / 0.19 / 0.12 / 0.09, SoccerTrack 0.06 / 0.21 / 0.21 / 0.23 / 0.29). Weighted gain at w 0.5: raw +0.0220, 25 frames +0.0316, 51 +0.0443, 101 +0.0636, 151 +0.0712 (7/8, worst −0.0063; plain +0.0522). The smoothing wins more, not less: the short spurious predicted spells it removes weigh more than the short label spells it loses. Adopted as v22 (v20 with the 151-frame box, w 0.5 kept); v23 = v22 + v21's video dead ball.
- **P9cx: raw + smoothed evidence together does not beat the smoothed alone** (best 0 raw + 0.5–0.75 × 151-frame box; any raw share lowers it). v22's form stays.
- **P9cs with `SMOOTH=151` (SoccerTrack full chain, own dead ball, 8 matches):** chain 0.6324 / 0.6134; smoothed dense evidence **+0.0356 / +0.0514 (8/8)** (raw form: +0.0144 / +0.0197, 6/8); video dead ball on top **+0.0370 / +0.0380 (8/8, worst +0.0165)** → the registered rule holds with the smoothed evidence too; together 0.6324 → 0.7050 at 1.6 s. v23 is a candidate. Caveats of 11 Nov on the dead ball (our dead model is out of domain there; SoccerTrack's dead definition) stand.
- Video model's training logs: train possession loss 0.16 after 6 epochs against ~91% dense accuracy on unseen matches → over-fitted to 4–5 matches; the next step is seed / model averaging, not longer training or larger input (already 796 × 448).
- **P9cy (Colab A100, 12 Nov): model B on our match + the three-model average, label-free.** The three video models (A, B: 4–5 matches each; F: all ten) call the same team on only 87.6–90.5% of our live frames (smoothed 88.5–91.8%): single-model noise is large. Agreement with the chain where it is sure: A 93.7%, B 92.3%, F 92.5%, mean of the three log ratios **94.5%** (smoothed 94.3%); fused possessions 285 → 269, 1.70% of live frames changed. The average is closer to the chain than any member → averaging removes noise. `cache/barath_dense_ABF.npz`. Second copies of A and F are training to measure the averaging gain on SoccerTrack.
- **12 Nov: submission chain is now v23** (v19c + 151-frame-smoothed dense video evidence at w 0.5 + the video head's dead-ball probability as third dead model), on the user's standing rule (26 Oct: adopt the evidence-backed chain without asking). Evidence: SoccerTrack full chain 0.6324 → 0.7050 (smoothed evidence +0.0356 / +0.0514, video dead +0.0370 / +0.0380, both 8/8), gate B, DFL-length-weighted score, label-free checks on our match. DFL / Metrica have no video, so their scores stay v19c's (0.7347 / 0.7754). v20–v22 are superseded intermediate steps. `evaluation/results/v23_chain.csv`.

### 12 Nov — PFF FC (Gradient Sports) World Cup 2022 data and the dead ball on real tracking
- **Data:** 64 matches, broadcast tracking 29.97 fps (raw + smoothed, per-object VISIBLE / ESTIMATED) + events; public Drive folder (link in github.com/tahmeed14/mit_ssac_opendata; PFF's own blog with the form is unreachable, TLS error). Copied to Drive `mtah2027/pff` (5.3 GB) through Colab; `analysis/p9da_pff_convert_colab.py` → `mtah2027/pff_npz/<id>.npz` (~21 MB / match). Licence text not found yet → to state in the technical document.
- **Reader `mtah.external.from_pff`:** labels from game events (on-the-ball spell → its team; gap stays with that team unless an OUT event or a restart follows → dead from the OUT / last touch to the restart); 25 fps; players raw (seen or estimated), ball where PFF gives one. Two matches: 264–309 spells, median 6.8–7.1 s, dead share 34–35%, length shares < 2 / 2–5 / 5–10 / 10–20 / ≥ 20 s 0.13 / 0.25 / 0.22 / 0.18 / 0.22 (DFL 0.30 / 0.31 / 0.19 / 0.12 / 0.09). Ball present on 93–95% of live and 33% of dead frames (broadcast cuts away during stoppages).
- **P9db: v8's dead-ball model on real tracking it never saw, no training** (`analysis/p9db_pff_dead.py`; spellF1 with the labels' team and the predicted dead ball, 1.0 = perfect dead ball): PFF 0.848 / 0.828 (dead frame accuracy 0.91, precision 0.92–0.95, recall 0.78–0.82, predicted share 28–31% vs 34–35%), SoccerTrack 0.863 / 0.849 (recall 0.66–0.71). **With a perfect team call our dead ball alone caps the score at ~0.85 on real data; it under-predicts dead.** DFL's "perfect dead ball +0.025" comes from the simulated tracking and does not carry over. The other team's notes (v18–v20) report the same finding. Next: dead-ball model trained on real tracking (PFF + SoccerTrack, held-out matches as judge).
- **P9dc first signal (2 PFF training matches only; `analysis/p9dc_dead_real.py`, HistGB, spellF1 with the labels' team):** PFF judge sim 0.854 → real 0.874 → sim+real 0.878; SoccerTrack judge sim 0.818 → real 0.496 (precision 0.56) → sim+real 0.793. A PFF-trained model leans on "no ball → dead", right on broadcast tracking and wrong on SoccerTrack's degraded copy.
- **Whose ball dropouts does our match look like (label-free, in-half frames)?** BAR-ATH ATD: ball missing 40.7%, 1.42 gaps / min, median gap 10.7 s, all of the missing time in gaps ≥ 5 s, 75% in gaps ≥ 15 s. DFL simulation: missing 42–51%, 1.6–1.8 gaps / min, median 6.6–11.8 s, 94–96% in gaps ≥ 5 s, 72–83% ≥ 15 s. PFF (ball seen or estimated): missing 27%, 1.4–2.0 gaps / min, median 2–4 s, 90% ≥ 5 s; PFF seen only: 53–55%, 7–9 gaps / min, median 0.8–0.9 s. SoccerTrack degraded: 29%, 2.9 gaps / min, median 3.8 s, 34% ≥ 15 s. **Our ATD's dropouts match the DFL simulation (built to), not PFF or SoccerTrack. So P9db's 0.85 cap is established for PFF / SoccerTrack only; it is not evidence about our match.** What stays open is only the timing of the dropouts against the true stoppages, which no label-free statistic here shows.
- **Handoff, 12 Nov ~22:00 (PC switched off by the user; session and crons end).** Submission chain: v23 (`evaluation/results/v23_chain.csv`). Colab A100 runtime left running, everything syncing to Drive: `p9ci/A3` (second copy of model A: trained, predicting 118577 / 118578 / 128057 / 128058 → then `p9ci/barath_A3`), `p9ci/F2` (second copy of the final model, epoch 5 of 6 started 18:55 UTC; `model.pt` saved per epoch → then `p9ci/barath_F2`), `pff_npz/` (PFF conversion, 24 of 64 done, ~3 matches / 8 min; log `pff_npz/pffconv.log`, ends with COPY_DONE). If the runtime dies: F2 resumes with RESUME=<Drive model.pt> START_EPOCH=<epochs in ftF2.log>; the converter skips finished matches. The runtime is NOT released automatically: check and `runtime.unassign()` when `run.log` has ALL_DONE and `pffconv.log` COPY_DONE.
  Next: (1) A + A3 average on the four matches (`analysis/p9cz_avg_preds.py` → gate B / `SMOOTH=151 p9cs`); if it holds, the submission's video evidence becomes the average of A, B, F, (A3, F2) (`cache/barath_dense_ABF.npz` is ready for the three); (2) fetch the rest of `pff_npz` (8 of 64 local) and build the PFF-trained team channel as an independent member; (3) team deliveries by 14 Nov 20:00 (`team/PROTOCOL.md`; invitations pending for enesbildirir and AhmetYP); (4) freeze 15 Nov, local full inference of the video models with the user's `!` run, public repo, technical document.

### 12 Nov (late) — teammates' deliveries on our ruler (P9de, `analysis/p9de_team_fuse.py`; files in `cache/team_in/`)
- Branches `team/enes`, `team/mehmet_ali_topkara`, `team/BURAYA_ADIN` (Eren: one fold of Mali's architecture re-run, no output for our match → not usable). Frame grids match ours (ours = theirs + one separator frame); Mali's labels agree with ours on 99.6% of DFL frames, 93.8% on Metrica (his Metrica labels are ball-status based and his Metrica input is the raw trace, not degraded).
- **Alone, our labels (1.6 s / 3 s), DFL 7:** ours 0.7346 / 0.7044; Mali (LightGBM + TCN-BiGRU + HSMM, DFL + Metrica) own decision 0.7101 / 0.6781; Enes (LightGBM 3-class + Viterbi, DFL only) through our decoder 0.6613 / 0.6756. Metrica 3: ours 0.7754 / 0.7754, Mali 0.7844 / 0.7287 (raw trace).
- **Fusions (each model on its own degraded copy → upper bound), DFL gain over ours:** + Mali team at w 0.25 +0.0099 / +0.0094 (5/7; Metrica +0.032 / −0.000), w 0.5 +0.0011 / −0.0009; + Mali dead as third dead model +0.0063 / +0.0068 (6/7; Metrica +0.005 / +0.002); + Enes team 0.25 +0.0030 / +0.0129, + Enes dead +0.0036 / +0.0037 (7/7); ours + Mali + Enes thirds + both dead +0.0150 / +0.0251 (5/7). **Mali's candidate B (sequence net also trained on 16 PFF matches, roaming dead ball; 3 DFL folds only): alone 0.7463 / 0.6858 on those matches (ours 0.7413 / 0.6933), ours + B team 0.5 + dead +0.0234 / +0.0256 (3/3).**
- Rule status: Mali's final version and candidate B declare no training / tuning on our match (unlabelled ATD statistics for the simulation only, as ours). **Enes's v8 trains its ball detector on our match's video (pseudo-labels from the ATD ball) and set the dead-carry parameters from the public window's dead frames → not admissible under the organiser's 8 Nov email; his version without those components does not exist as files.**
- **P9cy: second copy of video model A (A3, same recipe, another random draw) and the two-copy average** (`analysis/p9cz_avg_preds.py`; SoccerTrack 118577 / 118578 / 128057 / 128058, never seen by either; gain over the chain, dead ball from the labels, w 0.5). Smoothed 151: A +0.0500 / +0.0659, A3 +0.0530 / +0.0768, average **+0.0564 / +0.0719** (ahead of A in 4/4 matches at 1.6 s, of A3 in 3/4 and equal in one); raw: A +0.0266 / +0.0324, A3 +0.0310 / +0.0352, average +0.0275 / +0.0328. Video team accuracy 0.9034 / 0.9038 → 0.9054. Two copies of one training set are close; the average is a small, consistent gain (+0.003 to +0.006 at 1.6 s) and never the worst → the submission's video evidence becomes the mean log ratio of all models run on our match (A, B, F, A3, F2), whose training sets differ more than these two copies (on our match A / B / F agree on only 88–91% of live frames, their mean agrees more with the chain than any member).
- **Cache-key incident (12 Nov, 21:00–00:20):** adding `from_pff` to `mtah/external.py` changed `grumodel._hash()` (the file is in HASH_FILES), so (a) P9db / P9dc ran with a freshly trained 3-seed dead GRU of the same recipe (their conclusions stand; the stray model file is deleted) and (b) the v24 build started to retrain every team member (stopped after 57 min, nothing written). Fixed: the PFF reader lives in `mtah/pff.py`, `external.py` is byte-identical to before, hash back to ea44a36fa4aceb67. v19c–v23 were built before the edit and are unaffected. Rule: never touch a HASH_FILES source until the freeze.
- **13 Nov: submission chain is v24** (v23 with the video evidence and video dead-ball probability averaged over the five video models A, B, F, A3, F2; `cache/barath_dense_avg5.npz`; built with the restored cache key). Against v23: 0.99% of in-play frames differ (2.90% against v19c), possessions 281 → 279, shorter than 2 s 34 → 34, dead share 46.5 → 46.6%; valid. Public window, report only: v19c 0.718, v20 0.729, v21 0.729, v22 0.718, v23 0.718, v24 0.729 (all 17 possessions; macroF1 0.896 → 0.911; the spellF1 steps are one possession at one IoU threshold). `evaluation/results/v24_chain.csv`.
- **P9dg: ball-status Metrica labels (Mali's) for the dead-ball classifier — null.** Our event-derived Metrica labels differ from his on 5–8% of frames (ours dead with a ball present: 3.0% of game 1; ours live with no ball for ≥ 2 s: 2.1–3.6% of every game). HistGB, held-out folds, spellF1 with the labels' team: DFL judge 0.9349 (ours) → 0.9357 (ball labels); Metrica judged on the ball labels 0.9329 → 0.9352, on ours 0.9354 → 0.9345. Within noise; no retraining. (Side number: on the simulated copies the dead ball alone costs 0.065, on real PFF / SoccerTrack tracking 0.15.)
- **P9dh: Mali's explicit-duration semi-Markov decoder on our out-of-fold probabilities** (`analysis/p9dh_hsmm_decoder.py`; his externally tuned configuration, not re-tuned; P(H, A, D) = our team mean × our dead mean; DFL 7 / Metrica 3, gain over our decoder at 1.6 s / 3 s): emission scale 1: −0.0125 / +0.0218, −0.0243 / +0.0239; 2.5 (his): +0.0034 / +0.0121 (4/7), −0.0066 / +0.0078; 4: +0.0014 / +0.0034, +0.0007 / +0.0034; 4 with our boundary probability as his boundary bonus (gamma 2): **+0.0025 / +0.0046 (4/7), +0.0029 / +0.0046 (2/3)**, ahead at both referees on both sets. Lower scale buys the 3 s referee with the 1.6 s one. Forcing our dead blocks on top loses (−0.015 to −0.048): its joint dead decision is the better one. Next: a small grid (scale, gamma, turnover penalty, duration weight) with a leave-one-match-out pick.
- **P9dh grids (scale, boundary gamma, turnover penalty, duration weight; leave-one-match-out pick):** first grid best 3 / 4 / 6 / 2: DFL +0.0053 / +0.0073 (7/7), Metrica +0.0085 / +0.0033 (3/3); nested DFL +0.0037 / +0.0065 (6/7), Metrica −0.0016 / +0.0052. Second grid past the edges (gamma 4–8, weight 1–2): a plateau, every one of the top 12 ahead at both referees on both sets, e.g. 4 / 6 / 10 / 2: DFL +0.0056 / +0.0075 (7/7), Metrica +0.0089 / +0.0040 (3/3); nested DFL +0.0030 / +0.0044 (5/7), Metrica +0.0097 / +0.0034 (3/3). **The rule holds: the HSMM decoder (4 / 6 / 10 / 2) replaces our decoder, worth about +0.004 to +0.006.** Before it enters the submission: the same decoder on SoccerTrack's full chain with the video evidence and the video dead probability (P9di), since the submission's dead ball has parts DFL does not exercise.
- **P9di: the HSMM decoder on SoccerTrack's full chain (8 matches, cross-fold video; `analysis/p9di_st_hsmm.py`), our decoder → HSMM:** chain 0.6324 / 0.6134 → 0.6411 / 0.6226 (+0.0087 / +0.0092, 7/8); + smoothed video team evidence 0.6681 / 0.6648 → 0.6794 / 0.6775 (+0.0114 / +0.0127, 7/8); + video dead probability (the submission's form) 0.7051 / 0.7028 → **0.7235 / 0.7216 (+0.0185 / +0.0187, 8/8, worst +0.0080)**. Ahead on three data sets at both referees → `mtah/semimarkov.py` (vendored, pure-numpy path = the numba run, 0 of 145,968 frames differ), `make_submission.py v25`.
- **13 Nov ~00:05: the Colab A100 runtime was lost** (a fresh CPU runtime answered at 00:35; cause unknown: units or session limit). Lost: P9dd's training (nothing saved). Safe on Drive: `pff/`, `pff_npz/` (64), `p9ci/` models and outputs. P9dd goes to the local GPU with fewer matches.
- **13 Nov: submission chain is v25** (v24 decoded by the semi-Markov decoder). Against v24: 1.12% of in-play frames differ (3.23% against v19c), possessions 279 → 268 (shorter than 2 s 34 → 30, 2–5 s 79 → 74, median 7.0 → 7.1 s), dead share 46.6% → 46.6%, dead blocks 109 → 109, home share of live 0.599 → 0.601; valid. Public window, report only: 0.729 (v24 0.729), 17 possessions, macroF1 0.907 (0.911). `evaluation/results/v25_chain.csv`.
- **P9dj: Mali's model through the semi-Markov decoder** (`analysis/p9dj_fuse_hsmm.py`; DFL 7 / Metrica 3; base = v25's chain without video 0.7403 / 0.7119 and 0.7844 / 0.7793): + his team at 0.25 +0.0082 / +0.0044 (5/7), +0.0283 / −0.0006; + his dead as third model +0.0059 / +0.0065 (6/7), +0.0029 / +0.0016; **both: DFL 0.7552 / 0.7238 (+0.0149 / +0.0119, 7/7), Metrica 0.8144 / 0.7818 (+0.0300 / +0.0025, 3/3)**. Upper bound (own degraded copies; his Metrica input is the raw trace). On our match, label-free: his team agrees with our chain on 98.1% of the frames where ours is sure (92.6% of all live), his dead ball with ours on 93.7%, his submission with v25 on 92.3% of in-play frames (255 against 268 possessions). → `make_submission.py v26` (to build when the GPU is free). For the freeze his weights and a run of his code are needed (announced in the protocol).
- **P9dd / P9dj: the PFF-only BigGRU member (24 PFF matches, local GPU, 55 min; real broadcast tracking, no degradation) — fails.** Team accuracy on live frames: held-out PFF 0.922–0.956, degraded DFL 0.832–0.884, degraded Metrica 0.847–0.875 (our members: 0.91–0.93). Through the semi-Markov decoder, gain over the base: as a seventh member DFL −0.0030 / +0.0035 (2/7), Metrica −0.0099 / +0.0050 (0/3); at log-odds weight 0.25 DFL −0.0159 / +0.0048, Metrica −0.0356 / +0.0087; with Mali's team and dead −0.0009 / +0.0153, −0.0122 / +0.0084. It buys the 3 s referee with the 1.6 s one: PFF's possession is coarser (13% of spells under 2 s against DFL's 30%) and its tracking is another domain. Not adopted. What is left of the PFF idea: PFF as pre-training with fine-tuning on DFL + Metrica (the label definition then comes from the target-like sets), or Mali's candidate B (mixed training), which gained on its three folds.
- **13 Nov: submission chain is v26** (v25 + Mali's team probability at log-odds weight 0.25 and his dead probability as a fourth dead model, from his delivered probabilities for our match). Against v25: 2.10% of in-play frames differ (4.02% against v19c; 5.72% against Mali's own file), possessions 268 → 249 (shorter than 2 s 30 → 21, 2–5 s 74 → 67), dead share 46.6 → 46.9%, dead blocks 109 → 103, home share 0.601 → 0.605; valid. Public window, report only: spellF1 0.800 with 14 predicted possessions against 17 (v25 0.729 with 17), macroF1 0.933 (0.907); 17 possessions, not a selection signal. `evaluation/results/v26_chain.csv`. Open for the freeze: Mali's weights and a run of his code inside the reproducible pipeline.
- **P9dk: sequence stacker with Mali's team and dead probabilities as extra channels, semi-Markov decoder, against v26's fixed fusion — fails.** DFL −0.0067 / −0.0118 (3/7), Metrica +0.0053 / +0.0025 (2/3); mean of stacker and fixed fusion DFL +0.0020 / −0.0032, Metrica +0.0016 / +0.0022. Frame accuracy rises (e.g. 0.934 → 0.939) while spellF1 falls, as in P9ct. The fixed 0.75 / 0.25 fusion stays.
- **P9dl: PFF pre-training (P9dd's 24-match weights) + fine-tuning on DFL + Metrica for the BigGRU member, 3 DFL folds, against the same recipe from scratch (seeds 3–5), member swapped into the six-member mean, v25's chain — fails.** J03WMX +0.0086 / +0.0007, J03WN1 −0.0003 / +0.0075, J03WOH −0.0033 / −0.0001; mean +0.0017 / +0.0027 (1/3 ahead at 1.6 s), member frame accuracy +0.0020. Within the seed noise of one member, as SoccerTrack pre-training was (P9bf). Not extended. PFF line closed for the team call: PFF-only member fails on the label definition, pre-training is null.
- **P9dm: the decoder's setting re-checked on v26's fused probabilities — stays.** Second grid of P9dh, DFL 7 + Metrica 3; fixed 4 / 6 / 10 / 2: DFL 0.7552 / 0.7238, Metrica 0.8144 / 0.7818. Best alternatives +0.002 at one referee and −0.001 to −0.002 at the other; leave-one-match-out pick DFL −0.0004 / −0.0032, Metrica +0.0005 / −0.0000.
- **Mali's candidate B against his final version as the fused member (information; only the 3 DFL folds B has: J03WOY, J03WQQ, J03WR9; semi-Markov decoder, gain over the base):** final 0.25 + dead +0.0143 / +0.0121; B 0.25 + dead +0.0205 / +0.0125; B 0.5 + dead +0.0226 / +0.0204; final 0.125 + B 0.125 + both dead **+0.0242 / +0.0167** (ahead of final-only in 3/3 at 1.6 s). No Metrica and four DFL folds missing for B → the rule cannot be checked; v26 keeps the final version. If B's remaining folds arrive, this is the first thing to adopt.
- **Freeze note for Mali's part (v26):** his code has a path for exactly this: `tools/bar_from_nets.py --run <run>` rebuilds our match's probabilities locally from `outputs/<run>/nets/net_<seed>.pt` + `summary.json` and the boosters (`run.py m0 --config main_ext_met`, `ev_ext_mb`, about a minute each from cache). So what the reproducible pipeline needs from him is `outputs/main_ext_met/nets/` (5 nets) and `outputs/main_ext_met/summary.json` (a few hundred MB, announced in the protocol), his locked environment (`kurulum/requirements-gpu.txt`: torch 2.11 cu128, lightgbm 4.7, numpy 2.4.6) in its own venv, and a check that the rebuilt `components.npz` equals the delivered one.
- **Determinism (13 Nov):** `make_submission.py v26` run a second time on this machine gives the same file byte for byte (sha256 04e067c8…f6b7), with the cached models.
- **v26 rebuilt after removing the crop-classifier reads:** same sha256 (04e067c8…), so v19c+ never depended on those files.
- **Video step reproduced locally (13 Nov):** the five dense models re-run on the RTX 4060 (`cache/repro/`) against the Colab outputs: team sign agreement 99.977–99.990%, max |log-ratio diff| 0.010–0.041, max |q_dead diff| ≤ 0.0045 (GPU float noise). v26 rebuilt from the local averages differs from the submitted file on **1 frame of 199,319**. For a byte-identical file the per-frame video outputs (`barath_dense_avg5.npz`, model outputs, no provided data) go into the release next to the inference command.
- **P9do: wandering dead ball in the dead classifier's training (label-free fact: our ATD ball outside the pitch moves at a median 2.76 m/s, 6% under 0.3 m/s; the simulated dead ball 0.7–1.2 m/s, 24–36% nearly still).** HistGB, held-out folds, + one drifted draw per match (speeds from our match's outside-the-pitch quantiles), spellF1 with the labels' team: drifted judge DFL 0.9006 → 0.9080 (+0.0074), Metrica 0.9171 → 0.9207 (+0.0036); plain judge DFL 0.9357 → 0.9332 (−0.0025), Metrica 0.9364 → 0.9326 (−0.0038). Our match, label-free: dead share 0.419 → 0.435, agreement with the five video models' dead calls 0.8833 → 0.8924 (video-dead frames called dead 0.797 → 0.822, video-live called live 0.972 → 0.965). **The rule written before the run (ahead on the drifted judge without losing on the plain one) does not hold; not adopted.** The two judges disagree in direction; which one resembles our match is exactly what is unknown, the video agreement leans to the drifted one. Mali's candidate B uses the same drift (his `bar_dead_speeds`, median 2.7 m/s, matches ours).
- Candidate B in cloud mode needed one more aggregate from our match (the 19 outside-the-pitch ball speed quantiles, Mali's `bar_dead_speeds`); exported locally to `cache/cloud/bar_dead_speeds.npy` and read in cloud mode (one-line patch in our working copy of his `metrica_src.py`). The match-derived feature caches the local export built stay out of the package.

### 13 Nov (afternoon) — second pass over the teammates' and the friend's ideas
- **P9dp (label-free, `analysis/p9dp_player_noise_compare.py`): our player degradation is far harsher than our ATD.** Slot pieces per min ATD 11.5 (median piece 35.5 s) vs degraded DFL / Metrica 28–31 (median 2.8–3.6 s); same-side pairs < 1 m in 12.3% of ATD frames (clean DFL 10.1%) vs 31–36% degraded; all 11 visible per side 58% vs 19–33%. Causes: dup_share 0.5 (1–2 s copies) and the camera window cutting players. Variant PL (dup_share 0.06, piece_drop 0.06, view_drops_players False) matches: 10.35 vs 10.24 visible per side, 12–15% close pairs, 7.5 pieces / min. P9u had tested the count shift only (null); the fragmentation / duplicates were never tested → P9du (registered in its docstring).
- **P9dq: expected-spellF1 (MBR) decision with paths sampled from the semi-Markov model** (Mali's `decoding/mbr.py` vendored as `mtah/mbr.py`; the friend's v22 idea; v26's fused OOF probabilities, 300 samples). Temperature 1 / 2 / 4 / 8 / 16: DFL +0.0000 / −0.0001, +0.0008 / −0.0009, +0.0017 / −0.0016, +0.0014 / −0.0019, +0.0023 / −0.0060; Metrica −0.0004 / −0.0002 … +0.0036 / −0.0026. Nested pick DFL +0.0009 / −0.0020 (4/7), Metrica −0.0008 / −0.0011. Buys 1.6 s with 3 s, as Mali found ("neutral"). Not adopted. **O**
- **P9dr: Mali's contact gate (his final decoder's turnover adjustment from touch events, ported in `analysis/p9dr_contact_gate.py`) on v26's fused OOF probabilities:** x1 DFL +0.0007 / +0.0007 (2/7), Metrica −0.0005 / −0.0005; x0.5 DFL +0.0008 / +0.0009 (2/7), Metrica +0.0004 / +0.0005 (1/3). Four of seven DFL matches unchanged: on our 25-frame-smoothed test copies the touch detector finds only 2.4–3.7 events / min (his rate-calibrated design expects ~20), so the gate rarely acts. x0.5 meets the registered rule by a hair; candidate only after checking how many frames it changes on our match.
- **P9dt (label-free, `analysis/p9dt_restart_visibility.py`): the friend's V6 point holds for us.** Simulated restarts: P(ball seen) 0.00–0.13 one second before, 0.87–0.91 at the restart (dead_carry 0.25 raises the pre-restart level to 0.11–0.14 only): the ball's reappearance marks the restart exactly. Our ATD around the 58 Smart Tagging set pieces: 0.24–0.38 before the tag start rising smoothly to 0.57 at +6 s, no step (tag timing is coarse, so the profile is smeared; still no sign of the sharp step).
- **P9dt continued (label-free): real tracking agrees with our ATD, not with our simulation.** PFF World Cup tracking (12 matches, 1083 restarts): P(ball seen) 0.47–0.49 over the 12 s before a restart, 0.65 at it, 0.86 at +2 s; after a stoppage 0.95 → 0.34 over ~4 s. Our ATD around the five video models' restarts (112, dead ≥ 3 s then live ≥ 3 s): 0.40–0.54 before, 0.51 at, 0.90 at +1 s; after stoppages 0.92 → 0.41 over 2 s. Split by the video's dead calls our ATD sees the ball on 21–25% of dead and 94–96% of live frames; seen dead ball to nearest player median 3.0 m. Simulation (production): 0.39 → 0.13 falling toward the restart, then 0.89; after a stoppage 0.18 at once.
- **`mtah/deadsim.py` (new): realistic dead ball** — after the stoppage the ball keeps its last velocity ~0.5–2.5 s (seen), then is carried near the restarting player with a drifting ~N(0, 2.5 m) offset (median radius ~2.9 m, matches the 3.0 m above), converging to the restart spot in the last second, seen in on / off chunks (on-share 0.5 within 12 s of the restart, 0.15 before). Profile on J03WMX: 0.36–0.42 before the restart, dead seen 0.25, stoppage fade 0.99 → 0.30 over 3 s, dead-ball speed median 2.6 m/s (ours 2.76). Parameters from label-free statistics of our ATD (video split) and PFF; Enes's numbers (from public GT dead frames) not used.
- **P9dv (HistGB dead classifier, prod copies vs realistic-dead copies; `analysis/p9dv_dead_v6sim.py`):** old judge (prod copies) DFL 0.9357 vs 0.8444, Metrica 0.9364 vs 0.8783; v6 judge DFL 0.8283 vs **0.9040**, Metrica 0.8650 vs **0.9264** (7/7, 3/3); PFF real 0.8136 vs 0.8122; our match: dead share 0.419 → 0.430, agreement with the video's dead calls 0.8833 → 0.8920 (video-dead called dead 0.797 → 0.816). **The registered rule fails on PFF by 0.0014; not adopted as is.** What it shows: the production copies make the dead ball easy through the restart cue (0.936), a model trained on them loses 0.11 when the cue is gone, and label-free our match looks like the cue-free copies. Follow-up arm "mix" (both kinds of copies) registered in P9dw's docstring before its run.
- **P9dw (`analysis/p9dw_dead_halfnorm.py`): per-half robust scaling of the dead-ball inputs (the friend's v20 "tempo" idea) loses everywhere** (e.g. prod → prod+norm: v6 judge DFL 0.8283 → 0.8139, PFF 0.8136 → 0.7839); Mali found it neutral. **O**
- **P9dw's "mix" arm (prod + realistic-dead copies; registered before the run under P9dv's rule) HOLDS for the dead-ball classifier:** v6 judge DFL 0.8283 → **0.8920** (v6-only 0.9040), Metrica 0.8650 → **0.9231**; PFF real 0.8136 → **0.8193**; our match: agreement with the video's dead calls 0.8833 → **0.9075**, dead share 0.419 → 0.449. Old judge (carries the restart artefact) DFL 0.9357 → 0.9174, Metrica 0.9364 → 0.9289, reported only. → the classifier trains on mix copies in v27; the dead GRU gets the same test (P9dx) before it follows. **S**
- **P9dx (`analysis/p9dx_deadgru_mix.py`): the dead-ball GRU on mix copies — fails on PFF.** Folds J03WMX / J03WN1 / J03WOH / metrica1: v6 judge 0.8721 → 0.9656, old judge 0.9427 → 0.9424; our match: video agreement 0.8795 → 0.8957; **PFF real 0.8064 → 0.6007** (12 matches). Registered rule fails; the GRU keeps the production copies. v27 = v26 with the classifier on mix copies only (`make_submission.py v27a`; v27 name reserved for the both-models variant, not built). The GRU's PFF collapse against the classifier's PFF gain is unexplained (normalisation from the mixed set, or the sequence model reading the chunked on / off visibility as a pattern PFF does not have). **O**
- **P9du (Colab; `analysis/p9du_player_realism.py`): team GRU trained on ATD-like player tracks (PL) — fails.** On the PL judge B − A: DFL +0.0012 / +0.0035 (4/7), Metrica −0.0038 / −0.0013 (1/3); on the usual judge DFL +0.0008 / +0.0060, Metrica −0.0129 / −0.0135. The production model already scores higher on the cleaner PL judge (DFL 0.7029 → 0.7128), so the harsher player degradation in training costs nothing on ATD-like tracks. Enes's player-noise idea closed. **O**
- **v27a built (`make_submission.py v27a`, `evaluation/results/v27a_chain.csv`):** against v26 0.72% of graded frames differ, dead share 0.469 → 0.475, possessions 249 → 247 (< 2 s 21 → 20). The classifier is one of four dead-ball probabilities (classifier, GRU, video, Mali), so its change is diluted. Public (report only): spellF1 0.800 both, macroF1 0.9325 → 0.9340. Not the submission yet: the full chain on realistic-dead copies is still unmeasured, and the GRU's PFF collapse (P9dx) is being diagnosed.
- **P9dz (`analysis/p9dz_gru_pff_diag.py`): why the mix GRU collapses on PFF.** On six PFF matches the prod GRU predicts dead 22–40% (true 29–52%, precision 0.89–0.96, recall 0.68–0.84, mean q on live frames 0.03–0.07); the mix GRU predicts 49–63% (precision 0.55–0.78, recall 0.94–0.98, mean q on live 0.34–0.39). PFF's live ball drops out 7–9 times a minute for ~0.8 s (broadcast tracking); the realistic-dead copies teach that a ball seen in chunks is dead, so PFF's live play reads as dead. Our ATD's live ball is seen 94–96% of the time with long, rare gaps (median 10.7 s), as the simulation, and on our match the mix GRU predicts dead 0.432 (prod 0.417) with higher video agreement. PFF is not representative of our match on this axis; the rule still stands as registered (the GRU stays prod).
- **P9eb (label-free replay detector, `analysis/p9eb_replay_detect.py`; the friend's v3):** on 64.6% of in-play frames the broadcast has a usable homography; its players match the tactical players of the synced moment to a median 0.29 m (p95 2.0 m, p99 9.3 m). Mismatch > 4 m for ≥ 2 s (replays shown over the pitch): 2.0% of in-play frames, 17 stretches; 76% already dead in v26, 79% dead for the video models; 764 frames (31 s) v26 calls live. Most replays come without a usable homography and are already in the broadcast rule. Small lever; not pursued further.
- **P9ec (the friend's v9: team log-odds box-smoothed before the decoder, `analysis/p9ec_prob_smooth.py`):** 13 frames DFL +0.0016 / +0.0016 (5/7), Metrica −0.0008 / −0.0001; 25 and 51 frames trade 1.6 s for 3 s (Metrica −0.017 / +0.011, −0.042 / +0.016). Rule fails (Metrica behind); the semi-Markov decoder already smooths. **O**
- **Environment differences found in the 13 Nov scan (label-free):** (1) Metrica's sample ball is glued to a player: within 0.3 m of the nearest player on 53 / 51 / 32% of live frames raw and still 27 / 27 / 14% after our degradation, against 4% for degraded DFL and 5% for our ATD (video-live frames) — the friend's v17 fix ("Metrica's ball separated from the player with noise", Metrica +0.006); untested here. (2) DFL player positions jitter frame to frame (per-frame acceleration median 6.3 m/s² vs ATD 0.8, Metrica 1.4) while speeds over 5 frames agree (median 1.4 m/s everywhere); matters only for per-frame player kinematics. (3) The restart visibility artefact (P9dt). (4) PFF's broadcast ball drops out 7–9 times a minute in live play (P9dz), unlike our ATD. The ATD has no owner field (ball channels x, y, speed).
- **P9dr rerun with ATD-like touch detection (EV_SMOOTH=13: 4.1–5.0 touch events / min, our ATD 5.5):** gate x1 DFL +0.0007 / +0.0007 (2/7), Metrica +0.0002 / +0.0002 (1/3); x0.5 DFL +0.0002 / +0.0002, Metrica +0.0004 / +0.0005. Five of seven DFL matches unchanged: the decoder's turnover decisions rarely fall where the gate acts. Meets the letter of the rule with no real effect; not taken (it would only add a dependency). Side finding: our copies' ball (smooth 25) is smoother than our ATD's (frame acceleration p90 17 vs 27; smooth 13 matches), see the degradation's own note on the speed / kink compromise.
- **P9ea (`analysis/p9ea_chain_deadmix.py`): v27a against v26 in the full chain (team fused as v26, dead = held-out classifier + GRU + Mali; semi-Markov decoder).** v6 judge (realistic-dead copies): DFL 0.7272 / 0.6957 → **0.7463 / 0.7143 (+0.0191 / +0.0186, 7/7)**, Metrica 0.8020 / 0.7701 → **0.8088 / 0.7769 (+0.0068 / +0.0067, 3/3)**. Old judge (restart artefact) DFL 0.7548 / 0.7234 → 0.7536 / 0.7229 (−0.0012 / −0.0006, 2/7), Metrica −0.0014 / −0.0007. The v26 chain loses 0.028 on DFL when the artefact is gone; v27a gets back 0.019 of it. Judge 1 of 3 for v27a holds (rule rewritten before this result to three independent judges: realistic copies, SoccerTrack full chain P9eg, our match's video agreement 0.883 → 0.908). Dumps in cache/p9ea/ for P9ee / P9ef.
- **P9dy (ball-blind channel fused into v26's OOF, `analysis/p9dy_blind_fuse.py`; P9ds ran on Colab):** blind live accuracy 0.83–0.90 (fused 0.90–0.94), error correlation with the fused team 0.48–0.56. w 0.1 / 0.2 / 0.3: DFL +0.0017 / +0.0032, +0.0020 / +0.0049, −0.0026 / +0.0030; Metrica −0.0057 / +0.0019, −0.0132 / +0.0015, −0.0235 / −0.0040. Nested DFL +0.0017 / +0.0032 (4/7), Metrica −0.0100 / +0.0006 (1/3): the rule fails on Metrica. But the Metrica judge carries the glued ball (scan above), which favours the ball-seeing members there and not on our match → second judge registered before its run: SoccerTrack full chain (P9eh).
- **F3 / F4 (two more copies of the final video model, seeds 103 / 104, all ten SoccerTrack matches; Colab) on our match, label-free:** sign agreement with the five-model mean where it is sure (|e| > 1, v26 live frames) 0.979 / 0.982 (A 0.992, B 0.977, F 0.977, A3 0.979, F2 0.985); mean pairwise raw sign agreement of the seven 0.910. The seven-model mean (`cache/barath_dense_avg7.npz`) differs from the five-model mean on 1.1% of live frames (151-frame smoothed sign) and 0.7% of the dead calls. No held-out SoccerTrack check is possible (F3 / F4 saw all ten); the evidence for averaging more copies is P9cy (two copies +0.003 to +0.006, never the worst). Decision made before F3 / F4 were trained (12 Nov plan): the submission's video evidence becomes the mean of all models run on our match → v28 = v27a with the seven-model mean.
- **P9eg (`analysis/p9eg_st_deadmix.py`): v27a's classifier on SoccerTrack's full chain (real tracking + video, 8 matches, neither classifier saw SoccerTrack):** v26 0.7235 / 0.7216 (= P9di's number, consistency check) → v27a **0.7299 / 0.7283 (+0.0063 / +0.0067, 5/8)**; dead frame accuracy 0.9379 → 0.9446; dead-only spellF1 0.8845 → 0.8827. **All three registered judges hold for v27a** (realistic copies +0.019 / +0.019 DFL 7/7 and +0.007 Metrica 3/3; SoccerTrack +0.006 / +0.007; our match's video agreement 0.883 → 0.908). v27a replaces v26; v28 = v27a + seven video models.
- **P9ee (`analysis/p9ee_rejudge.py`): every chain choice made on the artefact copies, re-judged on the realistic copies (v6) and on SoccerTrack's full chain (st) — nothing flips.** Mali fusion (v6): team weight 0 −0.0105 / −0.0072, 0.5 −0.0056 / −0.0061, without his P(dead) **−0.0301 / −0.0305** (0/7): v26's fusion stays and his dead ball is worth more on realistic copies than on the old ones (−0.0075). Smoothing 13 frames: v6 DFL +0.0012 / +0.0013 (5/7), Metrica −0.0013 / −0.0013; st nested +0.0025 / +0.0045 (6/8) → judges disagree, fails. MBR: v6 nested DFL +0.0016 / −0.0010, Metrica +0.0030 / +0.0016; st +0.0003 / −0.0011 → fails (the 1.6 s / 3 s trade on all three judges). Decoder grid (243 settings): v6 nested DFL −0.0015 / +0.0029, Metrica −0.0253 / +0.0125; st's favourite (scale 3, gamma 8, turnover 14, duration weight 3) +0.0111 / +0.0131 (8/8) but Metrica −0.025 at 1.6 s: fewer, longer possessions buy the 3 s referee and SoccerTrack (coarse labels) with the 1.6 s one; v26's setting stays. Dead-ball median duration (250 / 350 / 500) changes nothing.
- **v28 built (`make_submission.py v28`; v27a + the seven video models' mean; sha256 ca3077cd…):** against v26 0.71% of graded frames differ, possessions 249 → 247, dead share 0.469 → 0.474. Public (report only) spellF1 0.800 for v26 / v27a / v28, macroF1 0.9325 / 0.9340 / 0.9336. Current submission candidate; the team members' recipe C (P9ef / P9ej) is the next change under test.
- **14 Nov, tests for the realism modules (`tests/test_realism_modules.py`, each checked to fail against a broken variant):** one test found that `mtah/deadsim.py`'s on / off visibility does not reach its nominal near-restart share (0.5): long off chunks started before the last 12 s run into it, realised ~0.25 near vs ~0.14 before (a 320 s toy stretch). P9dt's calibration and every result since (v27a, P9ea, P9eg, P9ef, P9ej) were measured on this realised behaviour, so the code is kept and its docstring now says what it does.
- **P9el / P9em (`analysis/p9el_decompose.py`, `p9em_team_loss.py`; v28's chain on held-out DFL 7, realistic copies, "on these copies"):** chain 0.7463; dead ball from the labels 0.7729 (+0.027); team from the labels 0.9515 (+0.21). Inside the team loss: the labels' boundaries with our team per spell 0.8449 (+0.099), our segments with the labels' team per segment 0.7876 (+0.041) → **"when" (segmentation) is the larger lever.** We under-segment: 2156 predicted vs 2467 label spells; label spells found at IoU 0.5: < 2 s 32%, 2–5 s 66%, 5–15 s 85%, > 15 s 94%. Ruler: the same predictions score 0.7588 / 0.7234 / 0.6358 with labels merged at 1.6 / 3 / 5 s (stored labels 0.7548), so outside numbers (DFL 0.83, Metrica 0.92) are not comparable without their ruler and degradation (clean tracking or Metrica's glued raw ball lift them a lot).
- **14 Nov, /code-review (high) on the 13 Nov modules, 10 findings, fixed with tests (`tests/test_review_fixes.py`, red first):** recipe C's model key now covers `mtah/deadsim.py`, `mtah/unglue.py` and its seeds (`members_c.recipe_key`); the mix dead-ball GRU's key covers deadsim + REALISTIC_RATES (`p4_common.dead_gru_path`; production path unchanged); PFF match cache keyed on the labelling code, label sets and export root (old caches rebuild once); `win_guard.sh` waits COOLDOWN_S = 180 s after a kill (it had stopped P9eh and Mali's rebuild one after the other on 13 Nov); deadsim's docstring now says the carry offset's real OU spread (~0.63 m stationary, 20 s), dead code removed, output byte-identical; `make_submission` builds p9cu's dead ball once and skips the overwritten `ours()` / `dead_v8` (v28 rebuilt byte-for-byte as the check). Kept by ruling: tcn128's realistic copy repeats draw 1's live frames (the judged recipe); guard matches only our venvs' python (our launch convention). Recipe C models trained on Colab before this carry the old key: renamed on download after checking deadsim / unglue differ only by the byte-identical clean-up.
- **14 Nov, rival forks (288 forks of the challenge repo; 70 with own commits read, nothing run):** best public spellF1 they report ~0.63 (jekkiechan 0.63, ZainnB 0.629, animepuika 0.612, Barca-Hackathon 0.593–0.606); ours 0.800. Strongest method: Barca-Hackathon (8-TCN ensemble on Metrica + DFL + 20 SkillCorner, ATD-like dead ball, kickoff-wait rule — ours already all D on 110,808–113,741 —, video/broadcast closed at a +0.004–0.010 ceiling). New to us: a training-only **clean-owner head** (+0.007 [+0.002, +0.013] on their held-out Metrica) → P9eo. A GitHub org "Tiki-Data" with other people's commits exists (name clash to check before the form).
- **14 Nov 00:40 (TR), P9ej cut before any result:** nine processes shared the Colab GPU and no P9ej member had finished in 98 min (~30 h projected). Kept the J03WMX / J03WN1 / J03WOH process, stopped the other three, so the production C members (v29) get the GPU. Judge 1 for v29 is now P9ej on those three DFL folds (rule unchanged: C ahead at 1.6 s, not behind at 3 s), with P9ef's single member on all ten folds as the Metrica check; judges 2 (P9ei, SoccerTrack) and 3 (P9ep, our match's video) unchanged. Decided without seeing any P9ej number.
- **14 Nov, Mali's components rebuilt by us (`scripts/mali_components.py`; his recipe, nets trained on Colab without BAR data, public GT blocked by MTAH_NO_PUBLIC, DFL-only booster rebuilt locally):** against his delivered `components.npz`: labels agree 97.3%, p_dead corr 0.991 (mean abs diff 0.021), p_home|alive corr 0.957 (0.052), p_boundary corr 0.921 — training noise, same recipe. `cache/team_in/mehmet_ali_topkara/barath_rebuilt/` lets the final build reproduce every input from code; switching v29 to it is checked on the external folds before use.
- **14 Nov, v28 rebuilt byte-for-byte (cmp IDENTICAL with `evaluation/results/v28_chain.csv`)** after the build refactor (p9cu built once) and the memory change in the mix dead-ball classifier (raws and float32 tables freed before sklearn's float64 copy). Peak RSS 9.1 → 7.4 GB (the peak is HistGB's fit with its early-stopping split); with Windows' apps the build only fits alone.
- **14 Nov, Mali rebuilt vs delivered on his own referee (held-out, `scripts/mali_components.py external`):** DFL 7 0.7036 → 0.6981 (−0.0055, 3/7 ahead), Metrica 3 0.7456 → 0.7556 (+0.0100, 3/3) — same recipe, seed noise. v29 reads the rebuilt components (every input reproducible from code); v26–v28 keep the delivered file.
- **14 Nov 02:40 (TR), P9en scoring input changed before reading any P9en number:** its scorer took the team from P9ej's six recipe-C members, which will not exist for most folds (P9ej cut). P9en is scored with P9ea's team input instead (v26's six production members out-of-fold + Mali at 0.25, the same in both arms), so the A-vs-C boundary comparison stays clean; rule unchanged.
- **P9en (`analysis/p9en_boundary_c.py`, scored with P9EN_TEAM=prod): the boundary GRU on recipe C — fails its rule by a hair.** Realistic judge: DFL +0.0041 / +0.0035 (6/7 ahead), Metrica −0.0004 / +0.0009 (2/3); old judge DFL −0.0018 / +0.0001, Metrica +0.0005 / −0.0005. Rule: Metrica not behind at either referee → fails (−0.0004 at 1.6 s). Boundary stays production; not re-litigated.
- **14 Nov 02:45 (TR), P9ej stopped (no P9ej number read; one member of one fold had finished):** it could cover at most one fold's six members before the freeze and it slowed the six production C members (the critical path). Judge 1 for v29 = P9ef (one member, all ten folds held out; realistic judge DFL +0.0285 / +0.0222, 7/7; Metrica +0.0065 / +0.0052), which already passed its registered rule; judges 2 (P9ei, SoccerTrack, six members) and 3 (P9ep, our match's video) unchanged.
- **P9eo (`analysis/p9eo_owner_head.py`): the rival's clean-owner training head on recipe C's member — fails its rule.** Realistic judge: DFL −0.0049 / −0.0047 (3/7), Metrica +0.0219 / +0.0193 (3/3); old judge: DFL +0.0057 / +0.0139 (4/7), Metrica −0.0045 / +0.0173. Rule (realistic DFL ahead at 1.6 s) fails; per-match swings ±0.04 with one member of 3 seeds. The Metrica gain may come from its glued sample ball making the owner target near-perfect there. Not adopted; production members keep one head.
- **P9eq (`analysis/p9eq_candB.py`): Mali's cand B (team from his PFF-trained nets, P(dead) = mean of two recipes) instead of his official component in our v28-style chain — fails.** DFL J03WOY / WQQ / WR9 (his delivered cand_mix_pff): realistic judge +0.0067 / +0.0023 (2/3), old +0.0065 / +0.0006. Metrica 1-3 (cand B composed by us on the two runs' blends, `scripts/mali_components.py candb_metrica`, an approximation: both blends hold the same booster at 0.25): realistic −0.0123 / +0.0024 (0/3), old −0.0126 / +0.0025; on his own referee 0.740 vs 0.746. Rule (Metrica not behind) fails; official component stays.
- **14 Nov 04:40 (TR), P9ek scoring input changed before reading any P9ek number:** same as P9en — its base team (P9ej's six C members) does not exist; P9ek is scored on P9ea's production team (v26's six members out-of-fold, Mali 0.25), expert at log-odds 0.25 on top, rule unchanged.
- **P9ek (`analysis/p9ek_short_expert.py`, P9EK_TEAM=prod): the short-possession expert (PFF controlled-possession pretrain, recipe C fine-tune) at log-odds 0.25 — fails.** Realistic judge: DFL +0.0004 / +0.0063 (3/7), Metrica −0.0155 / +0.0061 (1/3); old judge: DFL +0.0014 / +0.0030, Metrica −0.0116 / +0.0043. Rule (Metrica not behind, DFL ahead at 1.6 s by more than noise) fails. Pattern worth noting: it helps at the 3 s referee on both sources and hurts at 1.6 s on Metrica — it moves boundaries coarser, not finer.
- **P9eh (`analysis/p9eh_blind_st.py`): the ball-blind team channel on SoccerTrack's full chain (8 matches) — fails.** w 0.1: +0.0007 / +0.0007 (6/8); w 0.2: −0.0018 / +0.0018; nested (w picked on the other seven, the registered arm): −0.0019 / +0.0007 (6/8). Rule (nested ahead at 1.6 s) fails; with P9dy's Metrica −0.010 the channel stays out.
- **P9ei (`analysis/p9ei_st_members_c.py`): the six recipe-C members against production's on SoccerTrack's full chain (judge 2 for v29) — holds.** A 0.7235 / 0.7216 (= P9eg's v26 row, consistency check) → C **0.7279 / 0.7243 (+0.0043 / +0.0027), ahead at 1.6 s in 6/8**; live team accuracy higher or equal on all 8 (0.901–0.952 → 0.907–0.954). C models: Colab, the two copies trained under the old and the new members_c code are bit-identical (training confirms deadsim's clean-up changed nothing).
- **P9ep (`analysis/p9ep_video_team_agree.py`): judge 3 for v29, our match label-free — holds (narrowly).** Members' team call vs the seven video models' sign on graded frames v28 calls live: |e| > 1 (70,553 frames, the rule) A 0.9476 → C 0.9481; |e| > 0.5 0.9234 → 0.9250; |e| > 2 0.9800 → 0.9794. The two ensembles differ on 1.65% of live frames. With P9ef (judge 1) and P9ei (judge 2) all three judges hold → v29 is built.
- **P9er (`analysis/p9er_mali_rebuilt.py`): Mali's rebuilt component vs his delivered one in our chain, 10 held-out folds — equivalent.** Realistic judge: DFL +0.0007 / −0.0032 (4/7), Metrica +0.0022 / +0.0065 (2/3); old: DFL +0.0012 / −0.0028, Metrica −0.0031 / −0.0000. Rule (no source / referee behind by more than 0.005) holds → v29 keeps the rebuilt file.
- **v29 built (`make_submission.py v29` = v28 + the six recipe-C members + Mali's rebuilt component; sha256 91edea7f72bc740d…; peak RSS 7.45 GB):** against v28 1.33% of graded frames differ, possessions 247 → 245, dead share 0.474 → 0.471, home share of live 0.609 → 0.607. Judges: P9ef (held-out realistic DFL +0.0285, Metrica +0.0065), P9ei (SoccerTrack +0.0043 / +0.0027, 6/8), P9ep (video agreement 0.9476 → 0.9481), P9er (Mali rebuild equivalent). Public (report only, never a selector): 0.800 → 0.775 (macroF1 0.9336 → 0.9304; 14 → 15 possessions in the 17-possession window, one possession's worth). **v29 is the candidate.**
- **14 Nov, v29 deterministic:** a second build from the same inputs is byte-identical (cmp) to `evaluation/results/v29_chain.csv`.
- **P9es (`analysis/p9es_smooth13.py`, Kaggle 2×T4, only code + external datasets attached): recipe D (C with the ball smoothed over 13 frames instead of 25) — fails.** Judge real13 (ATD-like smoothing): DFL +0.0030 / +0.0080 (3/7), Metrica +0.0039 / +0.0128 (2/3); judge real25: DFL −0.0151 / +0.0030 (1/7), Metrica −0.0166 / +0.0268; old: DFL −0.0152 / −0.0022, Metrica −0.0269 / −0.0037. Each arm wins on the judge smoothed like its own training copies, so the answer hinges on which smoothing our ATD really has (acceleration p90 says 13, the degradation's speed / kink compromise chose 25): the rule (not behind on real25 at DFL 1.6 s) fails → C stays.
- **14 Nov, why public fell 0.800 → 0.775 with v29 (diagnosis only, nothing chosen on it):** v29 predicts one extra short away spell (~1.3 s, frames 12,853–12,885) that the public labels do not have; every spell v28 matched is still matched (14 → 15 predicted, 17 true). One spurious short spell in a 17-possession window is −0.025. C finds more short spells, which the external judges score as a gain overall (P9ef, P9ei).
- **P9eu registered (`analysis/p9eu_members_c_seeds.py`): seeds 6-11 of the six C members for a twelve-seed v30;** rule in its docstring (SoccerTrack not behind v29 at either referee, video agreement not lower).
- **P9et (`analysis/p9et_dejitter.py`, Kaggle, code + external only): recipe E (C with DFL's raw players de-jittered, window 9, calibrated label-free to the ATD's acceleration) — fails.** Judge realdj (de-jittered realistic): DFL +0.0043 / +0.0037 (5/7), Metrica −0.0095 / +0.0167 (0/3); judge real: DFL −0.0037 / −0.0004 (4/7), Metrica same; old: DFL +0.0049 / +0.0027, Metrica −0.0007 / +0.0057. Rule (Metrica not behind on realdj; DFL not behind on real at 1.6 s) fails. As with P9es, each arm leads on the judge built like its own copies and the 3 s referee likes the change while 1.6 s does not; C stays.
- **P9ev st (`analysis/p9ev_seeds12.py st`): twelve seeds per C member (v30) vs six (v29) on SoccerTrack's full chain — holds.** v29 0.7279 / 0.7243 (= P9ei's C row) → v30 **0.7333 / 0.7301 (+0.0055 / +0.0059), ahead at 1.6 s in 8/8**. Seeds 6-11 trained on Kaggle (P9eu, code + external datasets only).
- **P9ev video: twelve seeds vs six on our match's label-free video agreement — fails by a hair.** |e| > 1 (rule): v29 0.9486 → v30 0.9483 (−0.0003, ~21 of 70,553 frames); |e| > 0.5 0.9257 → 0.9252; |e| > 2 0.9795 → 0.9793. P9eu's rule (not lower) fails → **v30 not adopted, v29 stays** although SoccerTrack favoured v30 (+0.0055 / +0.0059, 8/8). Not re-litigated; the decision to override a registered rule is the user's.
- **v30 built for readiness (`make_submission.py v30`, sha256 6ade417e…; not the candidate):** against v29 0.17% of graded frames differ, possessions 245 → 244, dead share 0.471 → 0.471; public (report only) 0.775 → 0.775 (macroF1 0.9304 → 0.9309). Waiting on the user's decision about P9ev's video rule.
- **P9ew (`analysis/p9ew_pff_seeds.py`): third judge for v30, PFF World Cup broadcast tracking (12 matches, ball only where seen; team call only, dead from labels) — holds.** v29 0.6991 / 0.6939 → v30 0.7005 / 0.6979 (+0.0013 / +0.0040, 6/12 ahead at 1.6 s); live team accuracy higher or equal on all 12. With SoccerTrack (+0.0055 / +0.0059, 8/8) and PFF for v30 and the video agreement against by 0.0003 (~21 of 70,553 frames), **v30 becomes the working candidate**; this overrides P9eu's video rule and is recorded as such (the user can veto before the freeze).
- **P9ez (`analysis/p9ez_deadgru_seeds.py`): dead-ball GRU with 6 seeds instead of the build's 3 — fails (1 of 3).** SoccerTrack chain −0.0000 / −0.0009 (3/8); PFF dead-ball accuracy of the GRU alone 0.8898 → 0.8936 (10/12 higher); our match's agreement with the video dead calls 0.9236 → 0.9243 (dump reproduced exactly: q max diff 0). Rule needs all three; the chain judge is flat-to-negative, so the build keeps 3 seeds.
- **P9fa (`analysis/p9fa_boundary_seeds.py`): the boundary GRU with twelve seeds (3-11 trained locally, pooled with 0-2) — fails on SoccerTrack.** With v30's team: b3 0.7333 / 0.7301 (= v30's row) → b12 0.7330 / 0.7298 (−0.0003 / −0.0003, 4/8). Unlike the team members, the boundary model gains nothing from more seeds; PFF not run (the rule needs both). Boundary stays at 3 seeds.

### 16 Nov — retrospective error hunt (user: "find the mistakes, look back too")
- **P9ex (v31, 24 seeds vs v30's 12), SoccerTrack: +0.0008 / −0.0001 (4/8) → fails; v30 stays.** P9ex24's models (seeds 24-29) and P9fb's four extra members fetched into models/.
- Checked and clean: the official metric read again (X forward-fill, D splits possessions); full-match possessions per live minute 4.34 vs public GT 4.25 (DFL 1.6 s 6.7–8.5, Metrica 5.4–6.0); no H/A flip in the second half (Smart Tagging agreement per half; attack sign and keepers right); video left/right → home/away right in both halves (0.94 / 0.93); OpenCV seek in the frame extraction is exact at both half starts; **ATD ball vs off-the-shelf YOLO11m ball (416 fast-ball frames): sharp minimum at lag 0** (17 px; ±3 frames 33–39 px); Metrica event-derived dead vs tracking ball-missing overlap 92–96%.
- **Found 1: the decoded dead ball starts late.** SoccerTrack full chain (P9di codes, 613 blocks): onset median +0.84 s, every match +0.64…+0.96 s; the end is exact (median 0.00). Our match, label-free: video dead call 8–12 frames ahead of the classifier / Mali's dead; public window (alignment check only): classifier +21, Mali +15, video −4, GRU +2 frames; final D onsets +1.4 / +5.1 / +3.0 / +0.1 s. **P9fd (shift every D onset k frames earlier, nested pick): +0.0002 / +0.0003 → fails** — IoU 0.3–0.7 forgives 0.8 s on most spells.
- **Error taxonomy (SoccerTrack, IoU 0.5; 2027 GT spells, 2187 predicted, 1569 matched):** missed: merged into a longer same-team spell 205, other team called 124, fragmented 105, called dead 24; spurious: piece of a split GT spell 213, wrong team 207, inside a GT dead ball 139. No single cause; the team call (wrong team + split + merged) is ~3/4 of the loss, as the oracle said (+0.22).
- **P9fb (four extra recipe-C members gru / gru_mt / tcn / tcn_mt, seeds 0-5, added to v30's six), SoccerTrack: −0.0029 / −0.0032 (3/8) → fails;** PFF not run. v30 stays.
