# --- BLOK 1: Gerekli Kütüphanelerin Yüklenmesi ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import streamlit as st
import ast
import io

# --- BLOK 1.5: GLOBAAL SABİTLER ---
ROAS_COLS = ['ROAS 1', 'ROAS 3', 'ROAS 7', 'ROAS 14', 'ROAS 30', 'ROAS 60', 'ROAS 90']
ROAS_DAYS_NUMERIC = np.array([1, 3, 7, 14, 30, 60, 90])
ROAS_DAYS_LABELS = ['Gün 1', 'Gün 3', 'Gün 7', 'Gün 14', 'Gün 30', 'Gün 60', 'Gün 90']
COLOR_CYCLE = ['#FF0000', '#0000FF', '#FF8000', '#800080', '#A52A2A', '#00FFFF', '#FF00FF'] # Kırmızı, Mavi, Turuncu, Mor, Kahverengi, Cyan, Magenta

print("Kütüphaneler başarıyla yüklendi.")

# --- BLOK 2: MODEL KURMA (ÖNBELLEĞE ALINAN FONKSİYON) ---
@st.cache_data
def kur_modeli(file_buffer, original_filename):
    """
    Verilen CSV dosyasını okur ve 'p' (poly1d) model fonksiyonunu kurar.
    Sonucu (p fonksiyonunu) döndürür.
    """
    log_output = []
    try:
        file_buffer.seek(0) 
        data = pd.read_csv(file_buffer)
        log_output.append(f"'{original_filename}' başarıyla yüklendi (Önbellekten). {data.shape[0]} satır bulundu.")

        data.columns = data.columns.str.strip()
        
        historical_avg_roas = data[ROAS_COLS].mean().values
        model_params = np.polyfit(ROAS_DAYS_NUMERIC, historical_avg_roas, 3)
        p = np.poly1d(model_params)
        
        log_output.append("Curve-Fit Modeli (Önbellekten) başarıyla kuruldu.")
        
        return p, log_output

    except Exception as e:
        log_output.append(f"HATA (Model Kurulumu): {e}")
        return None, log_output

# --- BLOK 3 & 4: TAHMİN FONKSİYONU (GÜNCELLENDİ) ---
def calistir_tahmin(
    p_modeli,
    model_type, 
    pivot_day, 
    velocity_weights_str, 
    dampening_factor, 
    multi_roas_inputs_str, 
    baslangic_tarihi, 
    bitis_tarihi,
    save_directory # <-- 1. GÜNCELLEME: Parametre geri eklendi
):
    log_output = []
    fig = None 

    if p_modeli is None:
        log_output.append("HATA: Model ('p' fonksiyonu) kurulamadığı için tahmin yapılamıyor.")
        return None, log_output
        
    p = p_modeli
    
    try:
        # --- BLOK 4: GÖRSELLEŞTİRME (BAŞLANGIÇ) ---
        fig = plt.figure(figsize=(14, 9))
        
        # 1. Tarihsel Trend (Yeşil Çizgi)
        smooth_days = np.linspace(1, 90, 100) 
        smooth_roas = p(smooth_days) 
        plt.plot(smooth_days, smooth_roas, color='green', linestyle='-', linewidth=2, label='Tarihsel Trend Eğrisi (Tüm Veri)')

        for i in range(len(ROAS_DAYS_LABELS)):
            x_coord = ROAS_DAYS_NUMERIC[i]
            val_trend = p(x_coord)
            plt.annotate(f'{(val_trend * 100):.2f}%', (x_coord, val_trend), 
                         textcoords="offset points", xytext=(0, 7), 
                         ha='center', fontsize=8, color='green')

        # --- GİRDİLERİ PARÇALA ---
        try:
            VELOCITY_WEIGHTS = ast.literal_eval(velocity_weights_str)
            log_output.append(f"Velocity Ağırlıkları yüklendi: {VELOCITY_WEIGHTS}")
        except Exception as e:
            log_output.append(f"HATA: Velocity Ağırlıkları okunamadı. '{velocity_weights_str}' geçerli bir sözlük değil. Hata: {e}")
            return None, log_output

        try:
            MULTI_ROAS_INPUTS = ast.literal_eval(multi_roas_inputs_str)
            log_output.append(f"ROAS Girdileri yüklendi: {len(MULTI_ROAS_INPUTS)} kampanya bulundu.")
        except Exception as e:
            log_output.append(f"HATA: ROAS Girdileri okunamadı. '{multi_roas_inputs_str}' geçerli bir sözlük değil. Hata: {e}")
            return None, log_output

        
        # --- ANA TAHMİN DÖNGÜSÜ ---
        PIVOT_DAY_DYNAMIC = pivot_day # Pivot günü döngüden önce ayarla
        
        for i, (campaign_name, known_roas_inputs) in enumerate(MULTI_ROAS_INPUTS.items()):
            
            log_output.append(f"\n--- TAHMİN #{i+1}: {campaign_name} ---")
            
            MODEL_TYPE = model_type
            DAMPENING_FACTOR = dampening_factor

            pivot_value = known_roas_inputs.get(PIVOT_DAY_DYNAMIC)
            if pivot_value is None:
                log_output.append(f"HATA: {campaign_name} için Pivot Günü ({PIVOT_DAY_DYNAMIC}) verisi 'None'. Bu kampanya atlanıyor.")
                continue 
            
            log_output.append(f"Model Tipi: '{MODEL_TYPE}', Pivot Günü: d{PIVOT_DAY_DYNAMIC}, Girdi Değeri: {pivot_value:.4f}")
            prediction_days = [day for day in ROAS_DAYS_NUMERIC if day > PIVOT_DAY_DYNAMIC]

            velocity_ratio = 1.0
            
            if MODEL_TYPE == "velocity":
                log_output.append(f"Ağırlıklı Hız Hesabı (Pivot d{PIVOT_DAY_DYNAMIC}):")
                total_weighted_raw_ratio = 0.0
                total_weight = 0.0
                
                for base_day, weight in VELOCITY_WEIGHTS.items():
                    if base_day >= PIVOT_DAY_DYNAMIC:
                        continue
                    base_value = known_roas_inputs.get(base_day)
                    if base_value is None or base_value == 0:
                        continue

                    actual_velocity = pivot_value / base_value
                    historical_velocity = p(PIVOT_DAY_DYNAMIC) / p(base_day)
                    raw_velocity_ratio = actual_velocity / historical_velocity
                    
                    total_weighted_raw_ratio += raw_velocity_ratio * weight
                    total_weight += weight
                
                if total_weight > 0:
                    final_raw_velocity_ratio = total_weighted_raw_ratio / total_weight
                    velocity_ratio = 1 + ((final_raw_velocity_ratio - 1) * DAMPENING_FACTOR)
                    log_output.append(f"  > Sönümleme (Faktör {DAMPENING_FACTOR}): {velocity_ratio:.2f} (Ayarlı Fark {velocity_ratio-1:+.1%})")
                else:
                    log_output.append("  > Uyarı: Hız testi için yeterli veri yok. 'pivot' moda geçildi.")
            
            ideal_pivot_value = p(PIVOT_DAY_DYNAMIC)

            # Tahminler (kampanyaya özel)
            predictions = {}
            for day in prediction_days:
                historical_multiplier = p(day) / ideal_pivot_value
                growth_factor = historical_multiplier - 1
                adjusted_multiplier = 1 + (growth_factor * velocity_ratio)
                predictions[day] = pivot_value * adjusted_multiplier

            # Loglama (kampanyaya özel)
            model_name_str = "Ağırlıklı Hız" if velocity_ratio != 1.0 else "Dinamik Pivot"
            log_output.append(f"Sonuç Modeli: {model_name_str}")
            if velocity_ratio != 1.0:
                log_output.append(f"Hız Ayarı (Ağırlıklı): {velocity_ratio:.2f}x ({velocity_ratio-1:+.1%})")
            
            # --- DÖNGÜ İÇİ GRAFİK ÇİZİMİ ---
            graph_data_map = {}
            plot_days = []
            plot_values = []
            
            for day in ROAS_DAYS_NUMERIC:
                val = known_roas_inputs.get(day) if day <= PIVOT_DAY_DYNAMIC else predictions.get(day)
                graph_data_map[day] = val
                if val is not None:
                    plot_days.append(day)
                    plot_values.append(val)
            
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            
            plt.plot(plot_days, plot_values, marker='s', markersize=4, linestyle='--', color=color, label=campaign_name)

        # --- DÖNGÜ BİTTİ ---

        # --- BLOK 4: GÖRSELLEŞTİRME (FİNAL) ---
        plt.xscale('log')
        plt.xticks(ROAS_DAYS_NUMERIC, ROAS_DAYS_LABELS)
        plt.title(f'Tarihsel Trend vs. Çoklu Kampanya Tahmini (Log Eksen)', fontsize=16)
        plt.xlabel('ROAS Günü', fontsize=12)
        plt.ylabel('ROAS Değeri', fontsize=12)
        plt.grid(True, linestyle='--', which='both', alpha=0.6) 
        
        plt.text(0.70, 0.030, f"Tahmin Aralığı: {baslangic_tarihi} - {bitis_tarihi}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.axvline(x=PIVOT_DAY_DYNAMIC, color='gray', linestyle=':', label=f'Girdi/Tahmin Ayrımı (Gün {PIVOT_DAY_DYNAMIC})')
        
        plt.legend(loc='upper left') 
        log_output.append(f"\nGrafik başarıyla oluşturuldu. {len(MULTI_ROAS_INPUTS)} kampanya çizildi.")
        
        # --- 2. GÜNCELLEME: KAYDETME BLOĞU GERİ EKLENDİ ---
        if save_directory: # Kullanıcı bir yol girdiyse
            try:
                # Tarihleri dosya adı için güvenli hale getir
                safe_baslangic = baslangic_tarihi.replace('/', '-') if baslangic_tarihi else "TarihYok"
                safe_bitis = bitis_tarihi.replace('/', '-') if bitis_tarihi else "TarihYok"
                
                dinamik_dosya_adi = f"Multi_Tahmin_Pivot{PIVOT_DAY_DYNAMIC}_{safe_baslangic}_to_{safe_bitis}.png"
                
                # Dizinin var olduğundan emin ol
                os.makedirs(save_directory, exist_ok=True)
                save_path = os.path.join(save_directory, dinamik_dosya_adi)
                
                plt.savefig(save_path)
                log_output.append(f"Grafik başarıyla şu yola kaydedildi: {save_path}")
            
            except Exception as e:
                log_output.append(f"HATA (Grafik Kaydetme): {e}")
        # --- KAYDETME BLOĞU BİTTİ ---
        
    except Exception as e:
        log_output.append(f"HATA (Blok 3/4): Tahmin veya grafik oluşturulamadı: {e}")
        
    return fig, log_output


# --- BLOK 5: STREAMLIT ARAYÜZÜ (GÜNCELLENDİ) ---
def generate_auto_weights(pivot_day):
    """
    Seçilen pivot güne göre "Yakınlık Kuralı"nı kullanarak
    otomatik ağırlık sözlüğü oluşturan yardımcı fonksiyon.
    """
    base_days = [day for day in ROAS_DAYS_NUMERIC if day < pivot_day]
    
    if not base_days:
        return {}
        
    total_score = sum(base_days)
    
    weights_dict = {int(day): float(round(day / total_score, 4)) for day in base_days}
    
    return weights_dict

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("📈 Çoklu Senaryo ROAS Tahmin Aracı (Velocity Model)")
    
    DEFAULT_MULTI_ROAS_INPUTS = """{
    "Network1": {
        1: 0.0700, 3: 0.1200, 7: 0.1800, 14: null, 30: null, 60: null, 90: null
    },
    "Network2": {
        1: 0.0500, 3: 0.0900, 7: 0.1400, 14: null, 30: null, 60: null, 90: null
    },
    "Network3": {
        1: 0.0647, 3: 0.1012, 7: 0.1653, 14: null, 30: null, 60: null, 90: null
    },
    "Network4": {
        1: 0.0647, 3: 0.1012, 7: 0.1653, 14: null, 30: null, 60: null, 90: null
    },
    "Network5": {
        1: 0.0647, 3: 0.1012, 7: 0.1653, 14: null, 30: null, 60: null, 90: null
    }
}""".replace("null", "None")

    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Girdiler")
        
        uploaded_file = st.file_uploader("Tarihsel Veri CSV Dosyası (us11.csv)", type="csv")
        
        c1_1, c1_2 = st.columns(2)
        baslangic_tarihi = c1_1.text_input("Bölge (Opsiyonel)", "")
        bitis_tarihi = c1_2.text_input("Tarih Aralığı (Opsiyonel)", "")
        
        # --- 3. GÜNCELLEME: KAYIT YOLU METİN KUTUSU GERİ EKLENDİ ---
        save_directory = st.text_input("Grafik Kayıt Yolu (Opsiyonel)", "", help="Grafiğin kaydedileceği klasör yolu. Örn: /Users/onurkeklikscorp/tahmin")
        # --- GÜNCELLEME BİTTİ ---
        
        st.subheader("Kampanya ROAS Değerleri (Sözlük formatında)")
        st.info("Aşağıya istediğiniz kadar kampanya senaryosu ekleyebilirsiniz. Her kampanya adı eşsiz bir anahtar olmalıdır.")
        
        multi_roas_inputs_str = st.text_area(
            "Kampanya Veri Girdileri", 
            DEFAULT_MULTI_ROAS_INPUTS, 
            height=300, 
            label_visibility="collapsed"
        )

    with col2:
        st.header("2. Model Ayarları")
        
        model_type = st.selectbox("Model Tipi", ["velocity", "pivot"], index=0, help="`velocity` hızı dikkate alır, `pivot` sadece son noktayı alır.")
        
        pivot_day_options = [day for day in ROAS_DAYS_NUMERIC if day <= 30]
        pivot_day = st.selectbox("Pivot Günü (Son Veri Günü)", pivot_day_options, index=2)
        
        dampening_factor = st.slider("Sönümleme (Dampening) Faktörü", 0.0, 1.0, 1.0, 0.05, help="0.0 = Hız ayarı kapalı. 1.0 = Tam agresif (Varsayılan). 0.5 = Önerilen Denge.")
        
        st.subheader("Otomatik Hesaplanan Hız Ağırlıkları")
        st.info(f"`Pivot Günü` {pivot_day} olarak seçildi. Ağırlıklar 'Doğrusal Puanlama' ile otomatik hesaplandı.")
        
        auto_weights = generate_auto_weights(pivot_day)
        
        st.json(auto_weights) 
        
        velocity_weights_string_auto = str(auto_weights)

    st.divider()

    if st.button("🚀 Tahminleri Çalıştır", type="primary", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner('Model çalışıyor, lütfen bekleyin...'):
                
                file_buffer = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                p_modeli, model_log = kur_modeli(file_buffer, uploaded_file.name) 

                fig, tahmin_log = calistir_tahmin(
                    p_modeli=p_modeli,
                    model_type=model_type,
                    pivot_day=pivot_day,
                    velocity_weights_str=velocity_weights_string_auto,
                    dampening_factor=dampening_factor,
                    multi_roas_inputs_str=multi_roas_inputs_str,
                    baslangic_tarihi=baslangic_tarihi,
                    bitis_tarihi=bitis_tarihi,
                    save_directory=save_directory # <-- 4. GÜNCELLEME: Parametre fonksiyona geçirildi
                )
            
            st.header("3. Sonuçlar")
            
            full_log = model_log + tahmin_log
            
            out_col1, out_col2 = st.columns([1, 2])
            
            with out_col1:
                st.subheader("📝 Model Logları")
                st.text("\n".join(full_log))
                
            with out_col2:
                st.subheader("📊 Tahmin Grafiği")
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Grafik oluşturulamadı. Logları kontrol edin.")
        else:
            st.error("Lütfen bir tarihsel veri (CSV) dosyası yükleyin.")
