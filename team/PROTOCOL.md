# Tiki-Data — ortak model için teslim protokolü

GÖREV: More Than A Hack 2027 Phase 1 için bu projedeki possession modelinin çıktısını, Tiki-Data ekibinin ortak
modeline katılabilecek biçimde paketle ve ekip reposuna gönder. Yalnız aşağıdakiler isteniyor; toplamı yaklaşık
25 MB eder, 100 MB'ı geçmesin. Model ağırlığı, özellik, log, eski sürüm GÖNDERME.
Son tarih: 14 Kasım 2026, 20:00. Bir maddeyi veremiyorsan atlama; README'nin "Eksikler" başlığına nedeniyle yaz.

## Nereye
- Repo: github.com/MehdiSndg/mtah2027 (özel). Dalını `p5-env-cu128` dalından aç; dal adı: `team/<isim>`.
- Yalnızca `team/<isim>/` klasörünün içine yaz. Başka dosyaya dokunma, başka dala push etme, force-push yapma.
  Organizatörün reposuna (origin) asla push etme.
- Bağlantın zayıfsa: repoyu `git clone --depth 1 --branch p5-env-cu128 --single-branch` ile klonla (küçük iner).

## Asla koyma
- Organizatörün verdiği hiçbir veri: maç videosu, takip (ATD) dosyaları, Smart Tagging, etiketler (public GT dahil),
  bunlardan kesilmiş kare ya da ham satır. Yalnızca kendi kodun ve modelinin ürettiği çıktılar girer.
- Şifre, token, API anahtarı.

## 1) Bizim maç (BAR-ATH) için çıktılar -> `team/<isim>/barath/`
a) `submission.csv`: yarışmaya verdiğin son dosyanın aynısı (frame,label; label H/A/D/X).
b) `probs.npz` (`np.savez_compressed`, float16): kare kare olasılıklar, karar öncesi ham model çıktısı. Anahtarlar:
   - `p_home`, `p_away`, `p_dead`: şekil (N,), her karede toplamları 1
   - `valid`: bool, şekil (N,), modelin X verdiği (oyun dışı) karelerde False
   - N = örnek teslim dosyasındaki satır sayısı; i. eleman = frame sütunu i olan satır.
   - Model yalnızca etiket üretiyorsa (olasılık yoksa) bunu README'de açıkça yaz ve dosyayı 0/1 ile doldur.
c) Elinde hazır varsa, aynı kare ızgarasında `components.npz`: ölü top olasılığı (takım kararından önce), takım
   olasılığı (top oyundayken ev sahibi olasılığı), takım değişimi / sınır olasılığı. Yoksa üretme.

## 2) Dış veride doğrulama çıktıları -> `team/<isim>/external/`  (EN ÖNEMLİ MADDE)
Ağırlıklar bizim maçta ayarlanamaz (kural); o yüzden modelin EĞİTİMDE GÖRMEDİĞİ dış maçlardaki tahminleri şart.
Bunlar olmadan modele ağırlık verilemez. Her maç için bir dosya: `<veriseti>_<maç kimliği>.npz`
(`np.savez_compressed`), anahtarlar:
- `p_home`, `p_away`, `p_dead`: float16 (M,)
- `valid`: bool (M,)
- `labels`: int8 (M,), 0=H, 1=A, 2=D, 3=X (o veri setinin gerçek etiketi, senin kullandığın biçimde)
- `fps`: sayı
- `half`: int8 (M,), 1 ya da 2 (devre)
- `frame`: int32 (M,), her elemanın o veri setindeki özgün kare numarası (devre içinde, veri setinin kendi sayımıyla).
  Farklı kişilerin tahminlerini aynı kareye hizalamak için şart; kare atladıysan ya da yeniden örneklediysen de doldur.
- H = veri setinin kendi "home" takımı, A = "away" takımı. Takımları çevirdiysen README'ye yaz.

Kurallar:
- Tercih sırası: DFL açık verisinin 7 maçı (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), her maç sırayla
  dışarıda bırakılarak; sonra Metrica örnek maçları 1-3. Bunlar yoksa hangi dış veride doğruladıysan onu ver.
- Tahminler, bizim maça uygulanan modelle AYNI girdi koşulunda üretilmiş olmalı (ör. dış veriyi bizim maçın takip
  kalitesine benzetmek için bozduysan bozulmuş haliyle). Koşulu README'ye yaz.
- Bir maç eğitimde görüldüyse o maçın dosyasını KOYMA.
- DFL'in 7 maçında görülmemiş-maç tahminin yoksa ama 1-2 saatte üretebiliyorsan ÜRET: herkesin aynı maçlarda tahmini
  olursa modeller birlikte ağırlıklandırılabilir; yoksa yalnız tek tek skora göre kaba ağırlık verilebilir.

## 3) Birden fazla sürüm
Yalnız son sürüm zorunlu. Son sürümden yapıca farklı bir sürümün varsa (başka girdi, başka model ailesi, başka eğitim
verisi ya da başka karar yöntemi) ve çıktıları elinde hazırsa, onun 1 ve 2'deki dosyalarını `barath/<sürüm>/` ve
`external/<sürüm>/` altına ekle. Yeniden eğitim gerektiriyorsa ekleme; README'de adını an.

## 4) Kod -> `team/<isim>/code/`
Bizim maç için çıktıyı üreten kod (eğitim + çıkarım + karar / sonişlem) ve çalıştırma komutları. Yalnız kaynak
dosyalar; veri, ağırlık, önbellek yok.

## 5) `team/<isim>/README.md` (başlıkların hepsi, sırayla, kısa)
1. Yöntem özeti: 10-15 satır; girdi ne, hangi modeller, karar nasıl veriliyor.
2. Kullanılan veriler: her dış veri setinin adı, lisansı, kaç maç, eğitimde mi doğrulamada mı kullanıldı.
3. Bizim maçtan kullanılan girdiler: video / ATD / Smart Tagging hangileri, her biri nerede kullanılıyor.
4. KURAL BEYANI, her birine evet / hayır ve bir cümle:
   - Bizim maçın etiketleri (public 5 dakika dahil) eğitim, ayar ya da sürüm seçiminde kullanıldı mı?
   - Bizim maçın videosu ya da takip verisi üzerinde herhangi bir model eğitildi / ince ayar yapıldı / kalibre edildi
     mi (kendi kendine etiketleme, sözde etiket dahil)?
   - Bizim maç için elle etiket ya da insan yapımı veri üretildi mi?
   - Çıktı videoya bakılarak kontrol edilip düzeltildi mi?
   - Sağlanan veri bir yapay zekâ servisine yüklendi mi?
   - Eşik, ağırlık gibi her ayar hangi veride seçildi?
   "Evet" olan bir madde varsa hangi bileşende olduğunu ve o bileşen olmadan çıktı üretilebilir mi, yaz.
5. Skorlar, tablo: sürüm | hangi veride | hangi maçlar | görülmemiş maç mı? | spellF1. Public 5 dakika skorunu ayrı
   satırda ver ve "public" diye işaretle. Skoru hangi betikle hesapladığını yaz.
6. Etiket tanımı: dış verideki "possession" etiketini nasıl ürettin / dönüştürdün (kısa aralar birleştirildi mi, kaç
   saniye, ölü top nasıl tanımlandı).
7. Sürüm geçmişi ve denenip tutmayanlar: kısa liste.
8. Zamanlama: bizim maçta kare numarası nasıl elde edildi, hangi videoya göre hizalı.
9. Eksikler: bu protokolde istenip veremediğin her madde ve NEDENİ (yok / üretilmedi / silindi / yeniden üretmek çok
   uzun / bağlantı yetmedi). Hiçbir madde sessizce atlanamaz.
10. Elinde olup göndermediğin, işe yarayabilecek şeyler (ağırlıklar, başka sürümler, ara çıktılar): bir satırla adı ve
    boyutu. Modelin ortak modele girerse, son teslimin tekrar üretilebilmesi için ağırlıkları 15 Kasım'da ayrıca
    istenecek; silme, hazır tut.
11. Soru için kime ulaşılır.

## 6) Göndermeden önce kendin doğrula, sonucu README'nin sonuna yaz
- `probs.npz`: N doğru mu, NaN var mı, toplamlar 1 mi, argmax ile `submission.csv` etiketleri kaç karede aynı (yüzde).
- `external/` dosyaları: her dosyada dizilerin uzunluğu eşit mi, labels dağılımı (H/A/D/X yüzdeleri).
- Tablodaki skor, `external/` dosyalarından yeniden hesaplanınca aynı çıkıyor mu.
- `git status`: organizatör verisi yok, toplam boyut 100 MB altında.

## Son adım
Dalı push et ve tek mesajla bildir: dal adı, commit kimliği, README'nin "Eksikler" başlığı.
Emin olmadığın yerde tahmin yürütme; README'ye "emin değilim" diye yaz.
