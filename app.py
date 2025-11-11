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

# --- BLOK 3 & 4: TAHMİN FONKSİYONU ---
def calistir_tahmin(
    p_modeli,
    model_type, 
    pivot_day, 
    velocity_weights_str, 
    dampening_factor, 
    roas_inputs_str, 
    tahmin_bolgesi, 
    baslangic_tarihi, 
    bitis_tarihi, 
    save_directory
):
    # (Bu fonksiyonun içi, bir önceki kodla tamamen aynı,
    #  hiçbir değişiklik yapılmadı.)
    
    log_output = []
    fig = None 

    if p_modeli is None:
        log_output.append("HATA: Model ('p' fonksiyonu) kurulamadığı için tahmin yapılamıyor.")
        return None, log_output
        
    p = p_modeli
    
    try:
        try:
            VELOCITY_WEIGHTS = ast.literal_eval(velocity_weights_str)
            log_output.append(f"Velocity Ağırlıkları yüklendi: {VELOCITY_WEIGHTS}")
        except Exception as e:
            log_output.append(f"HATA: Velocity Ağırlıkları okunamadı. '{velocity_weights_str}' geçerli bir sözlük değil. Hata: {e}")
            return None, log_output

        try:
            known_roas_inputs = ast.literal_eval(roas_inputs_str)
            log_output.append(f"ROAS Girdileri yüklendi: {known_roas_inputs}")
        except Exception as e:
            log_output.append(f"HATA: ROAS Girdileri okunamadı. '{roas_inputs_str}' geçerli bir sözlük değil. Hata: {e}")
            return None, log_output

        log_output.append(f"\n--- YENİ TAHMİN (Ağırlıklı Hız Modeli) ---")
        
        MODEL_TYPE = model_type
        PIVOT_DAY_DYNAMIC = pivot_day
        DAMPENING_FACTOR = dampening_factor

        pivot_value = known_roas_inputs[PIVOT_DAY_DYNAMIC]
        if pivot_value is None:
            raise ValueError(f"PIVOT_DAY_DYNAMIC ({PIVOT_DAY_DYNAMIC}) için değer 'None'.")
        
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
                
                log_output.append(f"  > Hız Testi (d{base_day} -> d{PIVOT_DAY_DYNAMIC}) [Ağırlık: {weight:.0%}]")
                log_output.append(f"    - Gerçek Hız: {actual_velocity:.2f}x | Tarihsel Hız: {historical_velocity:.2f}x | Ham Oran: {raw_velocity_ratio:.2f}x")
            
            if total_weight > 0:
                final_raw_velocity_ratio = total_weighted_raw_ratio / total_weight
                velocity_ratio = 1 + ((final_raw_velocity_ratio - 1) * DAMPENING_FACTOR)
                log_output.append(f"  > Ağırlıklı Ortalama Ham Oran: {final_raw_velocity_ratio:.2f} (Ortalamadan {final_raw_velocity_ratio-1:+.1%})")
                log_output.append(f"  > Sönümleme (Faktör {DAMPENING_FACTOR}): {velocity_ratio:.2f} (Ayarlı Fark {velocity_ratio-1:+.1%})")
            else:
                MODEL_TYPE = "pivot"
        
        ideal_pivot_value = p(PIVOT_DAY_DYNAMIC)
        predictions = {}
        log_output.append(f"\nDinamik Katsayılar (d{PIVOT_DAY_DYNAMIC} bazlı, Hız Ayarlı):")
        
        for day in prediction_days:
            historical_multiplier = p(day) / ideal_pivot_value
            growth_factor = historical_multiplier - 1
            adjusted_multiplier = 1 + (growth_factor * velocity_ratio)
            predictions[day] = pivot_value * adjusted_multiplier
            log_output.append(f" d{day} Katsayıları: {historical_multiplier:.2f}x (Tarihsel) | Hız Ayarlı: {adjusted_multiplier:.2f}x")

        model_name_str = "Ağırlıklı Hız" if velocity_ratio != 1.0 else "Dinamik Pivot"
        log_output.append(f"\n--- DÖNEM TAHMINI SONUCU ({model_name_str}) ---")
        log_output.append(f"Tahmin Bölgesi: {tahmin_bolgesi}")
        log_output.append(f"Tahmin Aralığı: {baslangic_tarihi} - {bitis_tarihi}")
        log_output.append(f"------------------------------------")
        log_output.append(f"Girdi (ROAS {PIVOT_DAY_DYNAMIC}): {pivot_value:.4f} ({(pivot_value * 100):.2f}%)")
        if velocity_ratio != 1.0:
            log_output.append(f"Hız Ayarı (Ağırlıklı): {velocity_ratio:.2f}x ({velocity_ratio-1:+.1%})")
        log_output.append(f"------------------------------------")
        
        for day, pred_val in predictions.items():
            log_output.append(f"Tahmin Edilen ORTALAMA ROAS {day} Değeri: {pred_val:.4f} ({(pred_val * 100):.2f}%)")


        # --- BLOK 4: GÖRSELLEŞTİRME ---
        log_output.append("\n--- Birleşik Grafik Oluşturuluyor ---")

        fig = plt.figure(figsize=(14, 9))
        
        smooth_days = np.linspace(1, 90, 100) 
        smooth_roas = p(smooth_days) 
        
        graph_data_map = {}
        plot_days = []
        plot_values = []
        
        for day in ROAS_DAYS_NUMERIC:
            val = known_roas_inputs.get(day) if day <= PIVOT_DAY_DYNAMIC else predictions.get(day)
            graph_data_map[day] = val
            if val is not None:
                plot_days.append(day)
                plot_values.append(val)
        
        plt.plot(smooth_days, smooth_roas, color='green', linestyle='-', linewidth=2, label='Tarihsel Trend Eğrisi (Tüm Veri)')
        plt.plot(plot_days, plot_values, marker='s', linestyle='--', color='red', label=f'Tahmin Eğrisi ({model_name_str} d{PIVOT_DAY_DYNAMIC} Girdi ile)')
        
        for i in range(len(ROAS_DAYS_LABELS)):
            x_coord = ROAS_DAYS_NUMERIC[i]
            val_trend = p(x_coord)
            val_pred = graph_data_map.get(x_coord)
            
            trend_offset = (0, 7)
            pred_offset = (0, -15)
            
            if val_pred is not None:
                if val_trend < val_pred:
                    trend_offset = (0, -15)
                    pred_offset = (0, 7)
            
            plt.annotate(f'{(val_trend * 100):.2f}%', (x_coord, val_trend), textcoords="offset points", xytext=trend_offset, ha='center', fontsize=8, color='green')
            if val_pred is not None:
                plt.annotate(f'{(val_pred * 100):.2f}%', (x_coord, val_pred), textcoords="offset points", xytext=pred_offset, ha='center', fontsize=8, color='red')
        
        plt.xscale('log')
        plt.xticks(ROAS_DAYS_NUMERIC, ROAS_DAYS_LABELS)
        plt.title(f'Tarihsel Trend vs. {model_name_str} Tahmini (Log Eksen)', fontsize=16)
        plt.xlabel('ROAS Günü', fontsize=12)
        plt.ylabel('ROAS Değeri', fontsize=12)
        plt.grid(True, linestyle='--', which='both', alpha=0.6) 
        
        plt.text(0.735, 0.065, f"Tahmin Aralığı: {baslangic_tarihi} - {bitis_tarihi}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.text(0.82, 0.030, f"Bölge: {tahmin_bolgesi}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.axvline(x=PIVOT_DAY_DYNAMIC, color='gray', linestyle=':', label=f'Girdi/Tahmin Ayrımı (Gün {PIVOT_DAY_DYNAMIC})')
        
        if MODEL_TYPE == "velocity" and velocity_ratio != 1.0:
             plt.text(0.01, 0.88, f"Hız Ayarı (Ağırlıklı): {velocity_ratio:.2f}x ({velocity_ratio-1:+.1%})", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.legend(loc='upper left') 
        log_output.append(f"Grafik başarıyla oluşturuldu.")
        
    except Exception as e:
        log_output.append(f"HATA (Blok 3/4): Tahmin veya grafik oluşturulamadı: {e}")
        
    return fig, log_output


# --- BLOK 5: STREAMLIT ARAYÜZÜ ---
def generate_auto_weights(pivot_day):
    """
    Seçilen pivot güne göre "Yakınlık Kuralı"nı kullanarak
    otomatik ağırlık sözlüğü oluşturan yardımcı fonksiyon.
    """
    base_days = [day for day in ROAS_DAYS_NUMERIC if day < pivot_day]
    
    if not base_days:
        return {}
        
    total_score = sum(base_days)
    
    # --- BURASI DÜZELTİLDİ ---
    # Değerleri (value) numpy.float64 yerine standart python float'a çeviriyoruz.
    weights_dict = {int(day): float(round(day / total_score, 4)) for day in base_days}
    # --- DÜZELTME BİTTİ ---
    
    return weights_dict

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("📈 ROAS Tahmin Aracı (Velocity Model)")
    
    DEFAULT_ROAS_INPUTS = """{
    1: 0.0647,
    3: 0.1012,
    7: 0.1653,
    14: null,
    30: null,
    60: null,
    90: null
}""".replace("null", "None")

    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Girdiler")
        
        uploaded_file = st.file_uploader("Tarihsel Veri CSV Dosyası (m/d/y(dailydesc)-roas1-roas3-roas7-roas14-roas30-roas60-roas90)", type="csv")
        
        tahmin_bolgesi = st.text_input("Tahmin Bölgesi (Opsiyonel)", "")
        c1_1, c1_2 = st.columns(2)
        baslangic_tarihi = c1_1.text_input("Başlangıç Tarihi (Opsiyonel)", "")
        bitis_tarihi = c1_2.text_input("Bitiş Tarihi (Opsiyonel)", "")
        
        save_directory = st.text_input("Grafik Kayıt Yolu (Opsiyonel)", "")
        
        st.subheader("Bilinen ROAS Değerleri (known_roas_inputs)")
        roas_inputs_str = st.text_area("Bilinen ROAS Değerleri (known_roas_inputs)", DEFAULT_ROAS_INPUTS, height=220, label_visibility="collapsed")

    with col2:
        st.header("2. Model Ayarları")
        
        model_type = st.selectbox("Model Tipi", ["velocity", "pivot"], index=0, help="`velocity` hızı dikkate alır, `pivot` sadece son noktayı alır.")
        
        pivot_day_options = [day for day in ROAS_DAYS_NUMERIC if day <= 30]
        pivot_day = st.selectbox("Pivot Günü (Son Veri Günü)", pivot_day_options, index=2)
        
        dampening_factor = st.slider("Sönümleme (Dampening) Faktörü", 0.0, 1.0, 1.0, 0.05, help="0.0 = Hız ayarı kapalı. 1.0 = Tam agresif. 0.5 = Önerilen.")
        
        st.subheader("Otomatik Hesaplanan Hız Ağırlıkları")
        st.info(f"`Pivot Günü` {pivot_day} olarak seçildi. Ağırlıklar 'Doğrusal Puanlama' ile otomatik hesaplandı.")
        
        auto_weights = generate_auto_weights(pivot_day)
        
        # Düzeltilmiş sözlüğü (artık standart int ve float ile) göster
        st.json(auto_weights) 
        
        velocity_weights_string_auto = str(auto_weights)

    st.divider()

    if st.button("🚀 Tahmini Çalıştır", type="primary", use_container_width=True):
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
                    roas_inputs_str=roas_inputs_str,
                    tahmin_bolgesi=tahmin_bolgesi,
                    baslangic_tarihi=baslangic_tarihi,
                    bitis_tarihi=bitis_tarihi,
                    save_directory=save_directory
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
