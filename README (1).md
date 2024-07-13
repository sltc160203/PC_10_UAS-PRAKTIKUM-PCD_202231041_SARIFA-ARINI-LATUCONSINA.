
## UAS_PRAKTIKUM_PCD
## SARIFA ARINI LATUCONSINA (202231041)
## TEORI MENGENAI DETEKSI DAUN
Deteksi daun dalam pengolahan citra digital melibatkan beberapa langkah utama yang melibatkan teknik-teknik pemrosesan gambar. Berikut adalah teori singkat mengenai deteksi daun:

### 1. **Akuisisi Gambar**
Langkah pertama adalah memperoleh gambar dari daun yang akan dideteksi, biasanya menggunakan kamera digital atau sensor gambar.

### 2. **Preprocessing (Pra-pemrosesan)**
Pra-pemrosesan gambar dilakukan untuk meningkatkan kualitas gambar dan mempersiapkannya untuk analisis lebih lanjut. Teknik yang umum digunakan meliputi:
- **Konversi Warna**: Mengubah gambar dari ruang warna RGB (atau BGR dalam OpenCV) ke grayscale untuk menyederhanakan pemrosesan.
- **Peningkatan Kontras**: Menggunakan metode seperti histogram equalization untuk meningkatkan kontras gambar.

### 3. **Segmentasi Gambar**
Segmentasi adalah proses memisahkan objek (daun) dari latar belakang. Beberapa metode yang digunakan adalah:
- **Thresholding**: Membagi gambar berdasarkan tingkat kecerahan piksel. Metode ini sederhana namun efektif untuk gambar dengan latar belakang yang kontras.
- **Deteksi Tepi**: Menggunakan operator seperti Sobel, Canny, atau Laplacian untuk mendeteksi tepi objek dalam gambar.

### 4. **Ekstraksi Fitur**
Setelah segmentasi, fitur-fitur penting dari daun diekstraksi. Fitur ini bisa berupa:
- **Kontur**: Menggunakan algoritma seperti `cv2.findContours` untuk menemukan kontur daun.
- **Ciri-ciri Bentuk**: Ekstraksi ciri bentuk seperti panjang, lebar, area, dan perimeter daun.

### 5. **Klasifikasi**
Jika tujuan deteksi daun adalah identifikasi jenis daun, maka langkah klasifikasi dilakukan. Metode yang umum digunakan meliputi:
- **Klasifikasi Berbasis Ciri**: Menggunakan ciri-ciri yang diekstraksi untuk mengklasifikasikan jenis daun menggunakan algoritma seperti K-Nearest Neighbors (KNN), Support Vector Machines (SVM), atau Decision Trees.
- **Jaringan Saraf Tiruan**: Menggunakan deep learning, khususnya Convolutional Neural Networks (CNN), untuk klasifikasi yang lebih akurat dengan data gambar besar.

### 6. **Post-processing (Pasca-pemrosesan)**
Langkah ini melibatkan penyempurnaan hasil segmentasi atau klasifikasi. Misalnya, menghaluskan tepi objek atau menggabungkan hasil segmentasi yang terfragmentasi.

### 7. **Visualisasi Hasil**
Hasil akhir dari deteksi dan segmentasi daun divisualisasikan untuk analisis lebih lanjut atau untuk ditampilkan kepada pengguna. Teknik visualisasi meliputi penandaan daun yang terdeteksi pada gambar asli atau menampilkan gambar biner hasil segmentasi.

### Aplikasi
- **Agrikultur**: Untuk memantau kesehatan tanaman, mendeteksi penyakit, dan mengoptimalkan perawatan tanaman.
- **Penelitian Botani**: Untuk klasifikasi dan identifikasi spesies tanaman.
- **Industri Pertanian**: Untuk otomatisasi proses panen dan pemeliharaan tanaman.

Secara keseluruhan, deteksi daun dalam pengolahan citra digital merupakan kombinasi dari berbagai teknik pemrosesan gambar dan pembelajaran mesin untuk mencapai hasil yang akurat dan efisien.

- ## Referensi Jurnal : http://jti.aisyahuniversity.ac.id/index.php/AJIEE

## Tahapan Cara Penyelesaian Project Deteksi Daun

Berikut adalah penjelasan dari kode yang diberikan untuk deteksi daun:
Berikut adalah penjelasan per baris kode untuk setiap langkah dalam skrip:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
Baris ini mengimpor tiga pustaka yang diperlukan: 
- `cv2` untuk pengolahan citra menggunakan OpenCV.
- `numpy` untuk manipulasi array.
- `matplotlib.pyplot` untuk visualisasi citra.

```python
# 1. Memuat dan menampilkan citra asli
image_path = '/mnt/data/IMG-20240713-WA0006.jpg'
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
```
1. Mendefinisikan jalur gambar yang akan dimuat.
2. Membaca gambar dari jalur yang ditentukan menggunakan `cv2.imread`.
3. Mengkonversi gambar dari ruang warna BGR ke RGB menggunakan `cv2.cvtColor`.

```python
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Citra Asli')
plt.axis('off')
```
1. Membuat sebuah figure dengan ukuran 10x5 inci.
2. Membuat subplot pertama dari dua subplot dalam satu baris.
3. Menampilkan gambar asli menggunakan `plt.imshow`.
4. Menambahkan judul "Citra Asli".
5. Menonaktifkan sumbu gambar.

```python
# 2. Membuat mask untuk segmentasi mangga
# Konversi gambar ke ruang warna HSV
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
```
1. Mengkonversi gambar dari ruang warna RGB ke HSV (Hue, Saturation, Value) menggunakan `cv2.cvtColor`.

```python
# Definisikan rentang warna untuk mask mangga
lower_green_mango = np.array([30, 50, 50])
upper_green_mango = np.array([80, 255, 255])
```
1. Mendefinisikan rentang warna HSV yang lebih rendah untuk mangga hijau.
2. Mendefinisikan rentang warna HSV yang lebih tinggi untuk mangga hijau.

```python
# Membuat mask
mango_mask = cv2.inRange(hsv_image, lower_green_mango, upper_green_mango)
```
1. Membuat mask biner untuk mangga hijau dengan menggunakan `cv2.inRange`, yang menghasilkan gambar biner di mana piksel yang berada dalam rentang warna akan berwarna putih, dan yang lainnya akan berwarna hitam.

```python
plt.subplot(1, 2, 2)
plt.imshow(mango_mask, cmap='gray')
plt.title('Mask Mangga')
plt.axis('off')
plt.show()
```
1. Membuat subplot kedua dari dua subplot dalam satu baris.
2. Menampilkan mask mangga menggunakan `plt.imshow` dengan colormap 'gray'.
3. Menambahkan judul "Mask Mangga".
4. Menonaktifkan sumbu gambar.
5. Menampilkan plot dengan `plt.show`.

```python
# 3. Segmentasi mangga
mango_segment = cv2.bitwise_and(original_image, original_image, mask=mango_mask)
```
1. Melakukan segmentasi mangga dengan menerapkan mask pada gambar asli menggunakan `cv2.bitwise_and`, yang mempertahankan piksel yang berada dalam mask dan menghilangkan yang lainnya.

```python
plt.figure(figsize=(5, 5))
plt.imshow(mango_segment)
plt.title('Segmentasi Mangga')
plt.axis('off')
plt.show()
```
1. Membuat sebuah figure baru dengan ukuran 5x5 inci.
2. Menampilkan hasil segmentasi mangga menggunakan `plt.imshow`.
3. Menambahkan judul "Segmentasi Mangga".
4. Menonaktifkan sumbu gambar.
5. Menampilkan plot dengan `plt.show`.

```python
# 4. Segmentasi daun
# Definisikan rentang warna untuk mask daun
lower_green_leaf = np.array([40, 40, 40])
upper_green_leaf = np.array([90, 255, 255])
```
1. Mendefinisikan rentang warna HSV yang lebih rendah untuk daun hijau.
2. Mendefinisikan rentang warna HSV yang lebih tinggi untuk daun hijau.

```python
# Membuat mask
leaf_mask = cv2.inRange(hsv_image, lower_green_leaf, upper_green_leaf)
```
1. Membuat mask biner untuk daun hijau dengan menggunakan `cv2.inRange`, yang menghasilkan gambar biner di mana piksel yang berada dalam rentang warna akan berwarna putih, dan yang lainnya akan berwarna hitam.

```python
# Segmentasi daun
leaf_segment = cv2.bitwise_and(original_image, original_image, mask=leaf_mask)
```
1. Melakukan segmentasi daun dengan menerapkan mask pada gambar asli menggunakan `cv2.bitwise_and`, yang mempertahankan piksel yang berada dalam mask dan menghilangkan yang lainnya.

```python
plt.figure(figsize=(5, 5))
plt.imshow(leaf_segment)
plt.title('Segmentasi Daun')
plt.axis('off')
plt.show()
```
1. Membuat sebuah figure baru dengan ukuran 5x5 inci.
2. Menampilkan hasil segmentasi daun menggunakan `plt.imshow`.
3. Menambahkan judul "Segmentasi Daun".
4. Menonaktifkan sumbu gambar.
5. Menampilkan plot dengan `plt.show`.

Skrip ini mengilustrasikan bagaimana memuat gambar, membuat mask untuk segmentasi objek berdasarkan warna, dan menampilkan hasil segmentasi untuk mangga dan daun.

### Kesimpulan

Deteksi daun dalam pengolahan citra digital adalah proses yang melibatkan beberapa langkah utama untuk memisahkan dan mengidentifikasi daun dari gambar. Proses ini dimulai dengan akuisisi gambar dan diikuti oleh berbagai teknik pra-pemrosesan seperti konversi warna dan peningkatan kontras. Segmentasi gambar adalah langkah kunci yang memisahkan objek daun dari latar belakang menggunakan metode seperti thresholding dan deteksi tepi.

Setelah segmentasi, fitur-fitur penting dari daun diekstraksi untuk tujuan analisis lebih lanjut. Ekstraksi fitur ini mencakup identifikasi kontur dan pengukuran ciri bentuk daun. Jika diperlukan, klasifikasi dilakukan menggunakan algoritma pembelajaran mesin untuk mengidentifikasi jenis daun.

Teknik post-processing digunakan untuk menyempurnakan hasil segmentasi atau klasifikasi sebelum hasil akhirnya divisualisasikan. Aplikasi dari deteksi daun meliputi sektor agrikultur, penelitian botani, dan industri pertanian, yang semuanya diuntungkan dari kemampuan untuk memantau kesehatan tanaman, mendeteksi penyakit, dan mengoptimalkan perawatan tanaman.

Secara keseluruhan, deteksi daun menggunakan pengolahan citra digital menawarkan solusi efektif untuk berbagai aplikasi, memungkinkan analisis yang lebih akurat dan otomatisasi proses yang sebelumnya memerlukan intervensi manusia.