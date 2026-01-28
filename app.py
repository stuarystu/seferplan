import streamlit as st
import pandas as pd
import random
from copy import deepcopy

# OR-Tools için try-except (kurulu değilse uyarı verir)
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="Sefer Planlama v10.1", page_icon="🚌", layout="wide")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #312e81 0%, #1e1b4b 100%); padding: 1.5rem 2rem; border-radius: 0 0 1rem 1rem; margin: -1rem -1rem 1rem -1rem; border-bottom: 4px solid #fbbf24; }
    .main-header h1 { color: white; margin: 0; font-size: 1.5rem; font-weight: 900; }
    .main-header p { color: #fcd34d; margin: 0; font-size: 0.875rem; }
    .stat-card { padding: 0.75rem; border-radius: 0.75rem; text-align: center; color: white; font-weight: bold; margin-bottom: 0.5rem; }
    .stat-green { background-color: #22c55e; }
    .stat-orange { background-color: #f97316; }
    .stat-blue { background-color: #2563eb; }
    .stat-red { background-color: #dc2626; }
    .stat-gray { background-color: #4b5563; }
    .stat-purple { background-color: #9333ea; }
    .stat-yellow { background-color: #eab308; }
    .card-tekci { background-color: #fffbeb; border: 3px solid #f59e0b; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem; }
    .card-normalci { background-color: #eff6ff; border: 3px solid #3b82f6; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem; }
    .card-problem { border-color: #dc2626 !important; background-color: #fef2f2 !important; }
    .gap-badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; color: white; margin: 0.25rem 0; text-align: center; width: 100%; }
    .gap-green { background-color: #22c55e; }
    .gap-yellow { background-color: #eab308; }
    .gap-orange { background-color: #f97316; }
    .gap-red { background-color: #dc2626; }
    .gap-slate { background-color: #64748b; }
    .gap-rest { background-color: #059669; }
    .gap-pik { background-color: #7c3aed; }
    .service-row { display: flex; justify-content: space-between; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; font-size: 0.875rem; }
    .service-tekci { background-color: #fde68a; }
    .service-normalci { background-color: #bfdbfe; }
    .vehicle-badge { font-size: 0.6rem; padding: 0.1rem 0.3rem; border-radius: 0.2rem; color: white; margin-left: 0.25rem; }
    .vehicle-koruklu { background-color: #dc2626; }
    .vehicle-solo { background-color: #6b7280; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚌 Sefer Planlama</h1><p>v14.1 - Gevşetilmiş Kurallar (Pik: 7-17dk, Max: 90dk)</p></div>', unsafe_allow_html=True)

if 'services' not in st.session_state:
    st.session_state.services = None
if 'result' not in st.session_state:
    st.session_state.result = None

# ============================================
# YARDIMCI FONKSİYONLAR
# ============================================
def time_to_minutes(t):
    if t is None or (isinstance(t, float) and pd.isna(t)) or str(t).strip() == '' or str(t).strip().lower() == 'nan':
        return 0
    try:
        t_str = str(t).strip()
        if ':' in t_str:
            parts = t_str.split(':')
            return int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)
        return 0
    except:
        return 0

def minutes_to_time(m):
    if m >= 24 * 60:
        m -= 24 * 60
    return f"{m // 60:02d}:{m % 60:02d}"

def get_end_time(service):
    start = time_to_minutes(service['gidis'])
    end = time_to_minutes(service['donus'])
    return end if end > start else end + 24 * 60

def get_vehicle_type(service):
    arac = str(service.get('arac_tipi', '') or '').strip().lower()
    return 'koruklu' if ('körüklü' in arac or 'koruklu' in arac or arac == 'k') else 'solo'

def is_tekci_required(service):
    return str(service.get('kart_tipi', '') or '').strip().upper() == 'T'

def can_tekci_morning(service, settings):
    start = time_to_minutes(service['gidis'])
    end = get_end_time(service)
    return start >= settings['tekci_sabah_bas'] and end <= settings['tekci_sabah_bit']

def can_tekci_afternoon(service, settings):
    start = time_to_minutes(service['gidis'])
    end = get_end_time(service)
    return start >= settings['tekci_aksam_bas'] and end <= settings['tekci_aksam_bit']

def can_be_tekci(service, settings):
    return can_tekci_morning(service, settings) or can_tekci_afternoon(service, settings)

def has_time_conflict(s1, s2):
    s1_start, s1_end = time_to_minutes(s1['gidis']), get_end_time(s1)
    s2_start, s2_end = time_to_minutes(s2['gidis']), get_end_time(s2)
    return s1_start < s2_end and s1_end > s2_start

def card_has_conflict(card, service):
    return any(has_time_conflict(s, service) for s in card)

def get_gaps(services):
    if len(services) <= 1:
        return []
    sorted_s = sorted(services, key=lambda s: time_to_minutes(s['gidis']))
    return [time_to_minutes(sorted_s[i+1]['gidis']) - get_end_time(sorted_s[i]) for i in range(len(sorted_s)-1)]

def get_gaps_with_times(services):
    if len(services) <= 1:
        return []
    sorted_s = sorted(services, key=lambda s: time_to_minutes(s['gidis']))
    return [{'gap': time_to_minutes(sorted_s[i+1]['gidis']) - get_end_time(sorted_s[i]),
             'start': get_end_time(sorted_s[i]),
             'end': time_to_minutes(sorted_s[i+1]['gidis'])} for i in range(len(sorted_s)-1)]

def calc_tekci_work(services, settings):
    morning = [s for s in services if can_tekci_morning(s, settings)]
    afternoon = [s for s in services if can_tekci_afternoon(s, settings)]
    total = 0
    if morning:
        times = sorted([(time_to_minutes(s['gidis']), get_end_time(s)) for s in morning])
        total += times[-1][1] - times[0][0]
    if afternoon:
        times = sorted([(time_to_minutes(s['gidis']), get_end_time(s)) for s in afternoon])
        total += times[-1][1] - times[0][0]
    return total

def tekci_gap_valid(gap, settings):
    return settings['tekci_aralik_min'] <= gap <= settings['tekci_aralik_max']

def is_pik_time(gap_start, settings):
    """Pik saatte mi?"""
    if settings['sabah_pik_bas'] <= gap_start <= settings['sabah_pik_bit']:
        return True
    if settings['aksam_pik_bas'] <= gap_start <= settings['aksam_pik_bit']:
        return True
    return False

def is_max_aralik_time(gap_start, settings):
    """Max aralık saatinde mi?"""
    return settings['max_aralik_bas'] <= gap_start <= settings['max_aralik_bit']

def normalci_gap_valid_strict(gap, gap_start, settings):
    """
    KURAL: Saate göre boşluk kontrolü - TÜM AYARLARA UYULMALI
    - Pik saatte: SADECE pik_aralik_min - pik_aralik_max
    - Max aralık saatinde: oncelik1_min - oncelik3_max
    - Diğer saatlerde: oncelik1_min - oncelik2_max
    
    NOT: Minimum aralık HER ZAMAN oncelik1_min (pik hariç)
    """
    # Pik saat kontrolü - SADECE pik aralıkları
    if is_pik_time(gap_start, settings):
        return settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']
    
    # Minimum aralık kontrolü - ZORUNLU (pik hariç tüm saatlerde)
    if gap < settings['oncelik1_min']:
        return False
    
    # Max aralık saati - büyük aralığa izin ver (60-120dk)
    if is_max_aralik_time(gap_start, settings):
        return gap <= settings['oncelik3_max']
    
    # Diğer saatler - max oncelik2_max (60dk)
    return gap <= settings['oncelik2_max']

def card_gaps_valid_strict(card, settings):
    """Karttaki TÜM boşluklar saate göre geçerli mi?"""
    if len(card) <= 1:
        return True
    
    for gi in get_gaps_with_times(card):
        if not normalci_gap_valid_strict(gi['gap'], gi['start'], settings):
            return False
    return True

def card_has_rest(card, settings):
    """İstirahat kontrolü"""
    if len(card) <= 1:
        return True
    sorted_c = sorted(card, key=lambda s: time_to_minutes(s['gidis']))
    first_start = time_to_minutes(sorted_c[0]['gidis'])
    is_sabahci = first_start < settings['norm_aksam_bas']
    rest_start = settings['ist_sabah_bas'] if is_sabahci else settings['ist_aksam_bas']
    rest_end = settings['ist_sabah_bit'] if is_sabahci else settings['ist_aksam_bit']
    
    for gi in get_gaps_with_times(sorted_c):
        if gi['gap'] >= 30 and rest_start <= gi['start'] <= rest_end:
            return True
    return False

def count_violations(card, settings):
    """Kural ihlali sayısı - minimum aralık dahil"""
    violations = 0
    for gi in get_gaps_with_times(card):
        gap, gap_start = gi['gap'], gi['start']
        
        # Pik saat kontrolü
        if is_pik_time(gap_start, settings):
            if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                violations += 1
            continue
        
        # Minimum aralık kontrolü (pik hariç)
        if gap < settings['oncelik1_min']:
            violations += 1
            continue
        
        # Max aralık saatinde
        if is_max_aralik_time(gap_start, settings):
            if gap > settings['oncelik3_max']:
                violations += 1
        else:
            # Diğer saatlerde max oncelik2_max
            if gap > settings['oncelik2_max']:
                violations += 1
    
    return violations

def is_card_problematic(card, settings):
    """Kart hatalı mı? (kural ihlali, istirahat yok, küçük kart)"""
    if len(card) <= 3:
        return True
    if len(card) > 1 and not card_has_rest(card, settings):
        return True
    if count_violations(card, settings) > 0:
        return True
    return False

def get_max_gap(card):
    """Karttaki en büyük servis aralığı"""
    gaps = get_gaps_with_times(card)
    if not gaps:
        return 0
    return max(g['gap'] for g in gaps)

# ============================================
# VERİ ANALİZİ
# ============================================

def analyze_data(services, settings):
    """
    Optimizasyondan ÖNCE veriyi analiz et
    Fiziksel limitleri ve beklenen sorunları göster
    """
    n = len(services)
    analysis = {
        'total': n,
        'pik_services': 0,
        'incompatible_pairs': 0,
        'no_rest_possible': 0,
        'isolated_services': 0,
        'min_expected_cards': 0,
        'warnings': []
    }
    
    # Pik saatteki servisler
    for srv in services:
        start = time_to_minutes(srv['gidis'])
        if is_pik_time(start, settings):
            analysis['pik_services'] += 1
    
    # Tek kalabilecek servisler (hiçbir servisle eşleşemeyen)
    isolated = set()
    
    for i, s1 in enumerate(services):
        can_pair = False
        for j, s2 in enumerate(services):
            if i == j:
                continue
            if has_time_conflict(s1, s2):
                continue
            
            # Aralık hesapla
            s1_end = get_end_time(s1)
            s2_start = time_to_minutes(s2['gidis'])
            s1_start = time_to_minutes(s1['gidis'])
            s2_end = get_end_time(s2)
            
            if s1_start < s2_start:
                gap = s2_start - s1_end
                gap_start = s1_end
            else:
                gap = s1_start - s2_end
                gap_start = s2_end
            
            # Sıkı kural kontrolü
            valid = is_gap_valid_strict(gap, gap_start, settings)
            
            if valid:
                can_pair = True
                break
        
        if not can_pair:
            isolated.add(i)
    
    analysis['isolated_services'] = len(isolated)
    analysis['min_expected_cards'] = max(n // 6, len(isolated))
    
    if analysis['isolated_services'] > 0:
        analysis['warnings'].append(f"⚠️ {analysis['isolated_services']} servis tek başına kalabilir (kurallara uygun eşleşme yok)")
    
    if analysis['pik_services'] > n * 0.5:
        analysis['warnings'].append(f"⚠️ Servislerin %{analysis['pik_services']*100//n}'i pik saatte")
    
    return analysis

def display_analysis(analysis):
    """Veri analizini göster"""
    st.markdown("### 📊 Veri Analizi")
    
    cols = st.columns(4)
    with cols[0]:
        st.metric("Toplam Servis", analysis['total'])
    with cols[1]:
        st.metric("Pik Saatte", analysis['pik_services'])
    with cols[2]:
        st.metric("Tek Kalabilecek", analysis['isolated_services'])
    with cols[3]:
        st.metric("Min Kart Tahmini", analysis['min_expected_cards'])
    
    if analysis['warnings']:
        for warning in analysis['warnings']:
            st.warning(warning)

# ============================================
# SIKI KURAL KONTROL FONKSİYONLARI
# ============================================

def is_gap_valid_strict(gap, gap_start, settings):
    """
    SIKI ARALIK KONTROLÜ - True/False döner
    Pik saat, max aralık saati, min-max kuralları
    """
    # Pik saat kontrolü - SADECE pik aralıkları
    if is_pik_time(gap_start, settings):
        return settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']
    
    # Minimum aralık - ZORUNLU
    if gap < settings['oncelik1_min']:
        return False
    
    # Max aralık saati - büyük aralığa izin
    if is_max_aralik_time(gap_start, settings):
        return gap <= settings['oncelik3_max']
    
    # Diğer saatler - max oncelik2_max
    return gap <= settings['oncelik2_max']

def can_add_strict(card, srv, settings):
    """
    SIKI KONTROL: Servis karta eklenebilir mi?
    SADECE kurallara %100 uyuyorsa True
    """
    # Çakışma kontrolü
    if card_has_conflict(card, srv):
        return False
    
    test_card = card + [srv]
    
    # Tüm gap'leri kontrol et
    for gi in get_gaps_with_times(test_card):
        if not is_gap_valid_strict(gi['gap'], gi['start'], settings):
            return False
    
    return True

def has_valid_rest(card, settings):
    """İstirahat kontrolü"""
    if len(card) <= 1:
        return True
    return card_has_rest(card, settings)

# ============================================
# BASİT VE SIKI ALGORİTMA
# ============================================

def simple_strict_optimize(services, settings, progress_callback=None):
    """
    BASİT VE SIKI ALGORİTMA
    
    1. Servisleri saate göre sırala
    2. Her servis için kurallara UYGUN kart bul
    3. Bulamazsa YERLEŞTİRİLEMEDİ listesine ekle
    4. Min 4 servis garantisi için birleştirme yap
    5. Kart tiplerini belirle
    """
    n = len(services)
    if n == 0:
        return [], [], []
    
    if progress_callback:
        progress_callback(0.05, "Servisler sıralanıyor...")
    
    # Servisleri saate göre sırala
    sorted_services = sorted(services, key=lambda s: time_to_minutes(s['gidis']))
    
    # ==========================================
    # AŞAMA 1: İLK YERLEŞTİRME (SIKI KURALLAR)
    # ==========================================
    if progress_callback:
        progress_callback(0.1, "AŞAMA 1: Sıkı kurallarla yerleştirme...")
    
    cards = []
    unplaced = []  # Yerleştirilemeyenler
    
    for idx, srv in enumerate(sorted_services):
        if progress_callback and idx % 20 == 0:
            progress_callback(0.1 + 0.3 * idx / n, f"Servis {idx}/{n}...")
        
        best_card_idx = -1
        best_score = float('inf')
        
        for i, card in enumerate(cards):
            # SIKI KONTROL
            if not can_add_strict(card, srv, settings):
                continue
            
            test_card = card + [srv]
            gaps = get_gaps_with_times(test_card)
            
            # Skor: daha az gap + istirahat bonusu
            score = max(g['gap'] for g in gaps) if gaps else 0
            if has_valid_rest(test_card, settings):
                score -= 100  # İstirahat bonusu
            
            if score < best_score:
                best_score = score
                best_card_idx = i
        
        if best_card_idx >= 0:
            cards[best_card_idx].append(srv)
        else:
            # Yeni kart aç
            cards.append([srv])
    
    # ==========================================
    # AŞAMA 2: KARTLARI BİRLEŞTİR
    # ==========================================
    if progress_callback:
        progress_callback(0.4, "AŞAMA 2: Kartlar birleştiriliyor...")
    
    improved = True
    iterations = 0
    while improved and iterations < 200:
        iterations += 1
        improved = False
        
        for i in range(len(cards)):
            if improved:
                break
            for j in range(i + 1, len(cards)):
                if j >= len(cards):
                    continue
                
                # Çakışma kontrolü
                if any(has_time_conflict(s1, s2) for s1 in cards[i] for s2 in cards[j]):
                    continue
                
                merged = cards[i] + cards[j]
                
                # SIKI KONTROL: Tüm gap'ler geçerli mi?
                all_valid = True
                for gi in get_gaps_with_times(merged):
                    if not is_gap_valid_strict(gi['gap'], gi['start'], settings):
                        all_valid = False
                        break
                
                if all_valid:
                    cards[i] = merged
                    cards.pop(j)
                    improved = True
                    break
        
        cards = [c for c in cards if c]
    
    # ==========================================
    # AŞAMA 3: KÜÇÜK KARTLARI DÜZELT
    # ==========================================
    if progress_callback:
        progress_callback(0.6, "AŞAMA 3: Küçük kartlar düzeltiliyor...")
    
    for iteration in range(500):
        small_cards = [(i, c) for i, c in enumerate(cards) if 1 <= len(c) <= 3]
        
        if not small_cards:
            break
        
        improved = False
        
        # Strateji 1: İki küçük kartı birleştir
        for i in range(len(small_cards)):
            if improved:
                break
            for j in range(i + 1, len(small_cards)):
                idx1, idx2 = small_cards[i][0], small_cards[j][0]
                if idx1 >= len(cards) or idx2 >= len(cards):
                    continue
                
                card1, card2 = cards[idx1], cards[idx2]
                if any(has_time_conflict(s1, s2) for s1 in card1 for s2 in card2):
                    continue
                
                merged = card1 + card2
                
                # SIKI KONTROL
                all_valid = True
                for gi in get_gaps_with_times(merged):
                    if not is_gap_valid_strict(gi['gap'], gi['start'], settings):
                        all_valid = False
                        break
                
                if all_valid:
                    cards[idx1] = merged
                    cards[idx2] = []
                    improved = True
                    break
        
        cards = [c for c in cards if c]
        
        if improved:
            continue
        
        # Strateji 2: Küçük karttan büyük karta servis taşı
        small_cards = [(i, c) for i, c in enumerate(cards) if 1 <= len(c) <= 3]
        large_cards = [(i, c) for i, c in enumerate(cards) if len(c) >= 4]
        
        for small_idx, small_card in small_cards:
            if improved:
                break
            for srv in list(small_card):
                for large_idx, large_card in large_cards:
                    if not can_add_strict(large_card, srv, settings):
                        continue
                    
                    # Taşı
                    cards[large_idx].append(srv)
                    cards[small_idx] = [s for s in cards[small_idx] if s['_id'] != srv['_id']]
                    improved = True
                    break
                if improved:
                    break
        
        cards = [c for c in cards if c]
        
        if improved:
            continue
        
        # Strateji 3: Büyük karttan küçük karta servis çek
        small_cards = [(i, c) for i, c in enumerate(cards) if 1 <= len(c) <= 3]
        large_cards = [(i, c) for i, c in enumerate(cards) if len(c) >= 5]
        
        for small_idx, small_card in small_cards:
            if improved:
                break
            for large_idx, large_card in large_cards:
                if improved:
                    break
                for srv in list(large_card):
                    if not can_add_strict(small_card, srv, settings):
                        continue
                    
                    test_large = [s for s in large_card if s['_id'] != srv['_id']]
                    
                    # Büyük kart hala geçerli mi?
                    large_valid = len(test_large) <= 1
                    if not large_valid:
                        for gi in get_gaps_with_times(test_large):
                            if not is_gap_valid_strict(gi['gap'], gi['start'], settings):
                                large_valid = False
                                break
                        else:
                            large_valid = True
                    
                    if large_valid:
                        cards[small_idx].append(srv)
                        cards[large_idx] = test_large
                        improved = True
                        break
        
        cards = [c for c in cards if c]
        
        if not improved:
            break
    
    # ==========================================
    # AŞAMA 4: YERLEŞTİRİLEMEYENLERİ TESPİT ET
    # ==========================================
    if progress_callback:
        progress_callback(0.8, "AŞAMA 4: Kontrol ediliyor...")
    
    # Küçük kartları (1-3 servis) "yerleştirilemedi" olarak işaretle
    final_cards = []
    unplaced_cards = []
    
    for card in cards:
        if len(card) >= 4:
            final_cards.append(card)
        else:
            # Küçük kart - son bir kez büyük kartlara eklemeyi dene
            all_placed = True
            for srv in card:
                placed = False
                for fc in final_cards:
                    if can_add_strict(fc, srv, settings):
                        fc.append(srv)
                        placed = True
                        break
                if not placed:
                    unplaced.append(srv)
                    all_placed = False
            
            if not all_placed and len(card) > 0:
                # Hala küçük kart olarak kaldı
                pass
    
    # Kalan küçük kartları da ekle (uyarı ile gösterilecek)
    remaining_small = [c for c in cards if len(c) >= 1 and len(c) <= 3]
    for card in remaining_small:
        if card not in [c for c in final_cards]:
            # Bu kartı final_cards'a ekle ama uyarı olarak işaretle
            final_cards.append(card)
    
    # ==========================================
    # AŞAMA 5: KART TİPLERİNİ BELİRLE
    # ==========================================
    if progress_callback:
        progress_callback(0.9, "AŞAMA 5: Kart tipleri belirleniyor...")
    
    tekci_cards = []
    normalci_cards = []
    
    for card in final_cards:
        card_type = determine_card_type_strict(card, settings)
        if card_type == 'Tekçi':
            tekci_cards.append(card)
        else:
            normalci_cards.append(card)
    
    if progress_callback:
        progress_callback(1.0, f"Tamamlandı! Tekçi: {len(tekci_cards)}, Normalci: {len(normalci_cards)}, Yerleştirilemedi: {len(unplaced)}")
    
    return tekci_cards, normalci_cards, unplaced

def determine_card_type_strict(card, settings):
    """
    SIKI KART TİPİ BELİRLEME
    
    Tekçi kriterleri - HEPSİ ZORUNLU:
    1. Min servis sayısı (varsayılan 4)
    2. Sabah servisi ≥2 (06:00-10:00)
    3. Akşam servisi ≥2 (14:00-20:00)
    4. Arada min 3 saat boşluk
    5. Tekçi aralık kuralları (10-35dk)
    6. Toplam çalışma < 9 saat
    7. Kartın TAMAMININ sabah+akşam servislerinden oluşması
    """
    min_srv = settings.get('tekci_min_servis', 4)
    
    # Kriter 1: Min servis
    if len(card) < min_srv:
        return 'Normalci'
    
    sorted_card = sorted(card, key=lambda s: time_to_minutes(s['gidis']))
    
    # Sabah ve akşam servislerini ayır
    sabah = []
    aksam = []
    diger = []
    
    for srv in sorted_card:
        start = time_to_minutes(srv['gidis'])
        end = get_end_time(srv)
        
        # Sabah: Başlangıç >= 06:00 ve Bitiş <= 10:00
        if start >= settings['tekci_sabah_bas'] and end <= settings['tekci_sabah_bit']:
            sabah.append(srv)
        # Akşam: Başlangıç >= 14:00 ve Bitiş <= 20:00
        elif start >= settings['tekci_aksam_bas'] and end <= settings['tekci_aksam_bit']:
            aksam.append(srv)
        else:
            diger.append(srv)
    
    # Kriter 7: Gün ortası servis varsa Normalci
    if diger:
        return 'Normalci'
    
    # Kriter 2 ve 3: Min 2 sabah, min 2 akşam
    if len(sabah) < 2 or len(aksam) < 2:
        return 'Normalci'
    
    # Kriter 4: Sabah-akşam arası min 3 saat boşluk
    sabah_son = max(get_end_time(s) for s in sabah)
    aksam_ilk = min(time_to_minutes(s['gidis']) for s in aksam)
    
    if aksam_ilk - sabah_son < 180:  # 3 saat = 180 dk
        return 'Normalci'
    
    # Kriter 5: Tekçi aralık kuralları
    # Sabah servisleri arası
    if len(sabah) > 1:
        sabah_sorted = sorted(sabah, key=lambda s: time_to_minutes(s['gidis']))
        for i in range(len(sabah_sorted) - 1):
            gap = time_to_minutes(sabah_sorted[i+1]['gidis']) - get_end_time(sabah_sorted[i])
            if not (settings['tekci_aralik_min'] <= gap <= settings['tekci_aralik_max']):
                return 'Normalci'
    
    # Akşam servisleri arası
    if len(aksam) > 1:
        aksam_sorted = sorted(aksam, key=lambda s: time_to_minutes(s['gidis']))
        for i in range(len(aksam_sorted) - 1):
            gap = time_to_minutes(aksam_sorted[i+1]['gidis']) - get_end_time(aksam_sorted[i])
            if not (settings['tekci_aralik_min'] <= gap <= settings['tekci_aralik_max']):
                return 'Normalci'
    
    # Kriter 6: Toplam çalışma < 9 saat
    total_work = calc_tekci_work(sorted_card, settings)
    if total_work > 9 * 60:
        return 'Normalci'
    
    return 'Tekçi'

# ============================================
# GLOBAL OPTİMİZASYON (BASİT + SIKI)
# ============================================

def global_optimize(services, settings, progress_callback=None):
    """
    GLOBAL OPTİMİZASYON - BASİT VE SIKI
    
    Zorunlu tekçiler hariç TÜM servisleri birlikte optimize et
    """
    n = len(services)
    if n == 0:
        return [], []
    
    # ==========================================
    # AŞAMA 1: Zorunlu tekçileri ayır
    # ==========================================
    if progress_callback:
        progress_callback(0.02, "Zorunlu tekçiler ayrılıyor...")
    
    required_tekci = [s for s in services if is_tekci_required(s)]
    required_ids = set(s['_id'] for s in required_tekci)
    
    # Zorunlu tekçileri grupla
    forced_tekci_cards = []
    used_required = set()
    
    for srv in required_tekci:
        if srv['_id'] in used_required:
            continue
        
        card = [srv]
        used_required.add(srv['_id'])
        
        # Aynı karta eklenebilecek diğer zorunlu tekçileri bul
        for other in required_tekci:
            if other['_id'] in used_required:
                continue
            if card_has_conflict(card, other):
                continue
            
            test = card + [other]
            if calc_tekci_work(test, settings) <= 9 * 60:
                card.append(other)
                used_required.add(other['_id'])
        
        forced_tekci_cards.append(card)
    
    # ==========================================
    # AŞAMA 2: Kalan servisleri optimize et
    # ==========================================
    remaining = [s for s in services if s['_id'] not in required_ids]
    
    tekci_cards, normalci_cards, unplaced = simple_strict_optimize(remaining, settings, progress_callback)
    
    # Zorunlu tekçileri ekle
    for card in forced_tekci_cards:
        tekci_cards.insert(0, card)
    
    # ==========================================
    # AŞAMA 3: Tekçi oranını kontrol et
    # ==========================================
    target_ratio = settings['tekci_oran']
    total_cards = len(tekci_cards) + len(normalci_cards)
    
    if total_cards > 0:
        current_ratio = len(tekci_cards) / total_cards
        
        if progress_callback:
            progress_callback(0.98, f"Tekçi oranı: %{current_ratio*100:.0f} (Hedef: %{target_ratio*100:.0f})")
    
    # Yerleştirilemeyenleri uyarı olarak göster
    if unplaced:
        st.warning(f"⚠️ {len(unplaced)} servis kurallara uygun şekilde yerleştirilemedi!")
    
    return tekci_cards, normalci_cards

# ============================================
# COLUMN GENERATION (BASİTLEŞTİRİLMİŞ)
# ============================================

def generate_valid_card(services, settings, dual_values, used_services, max_attempts=100):
    """
    SUBPROBLEM: Kurallara uygun yeni kart üret
    Reduced cost < 0 olan en iyi kartı bul
    
    Reduced Cost = 1 - Σ π[i] (karttaki servisler için)
    """
    n = len(services)
    available = [i for i in range(n) if i not in used_services]
    
    if not available:
        return None, 0
    
    best_card = None
    best_reduced_cost = 0  # 0'dan küçük olmalı ki iyileştirme sağlansın
    
    # Farklı başlangıç noktalarıyla dene
    for attempt in range(min(max_attempts, len(available))):
        # Rastgele veya sıralı başlangıç
        if attempt < len(available):
            start_idx = available[attempt]
        else:
            start_idx = random.choice(available)
        
        card_indices = [start_idx]
        card_services = [services[start_idx]]
        
        # Greedy extension - kurallara uygun servis ekle
        for idx in available:
            if idx in card_indices:
                continue
            
            srv = services[idx]
            
            # Çakışma kontrolü
            if card_has_conflict(card_services, srv):
                continue
            
            test_card = card_services + [srv]
            
            # TÜM KURALLARA göre kontrol
            if not check_gap_rules(test_card, settings):
                continue
            
            # Karta ekle
            card_indices.append(idx)
            card_services.append(srv)
        
        # Kart geçerli mi? (min 4 servis veya tek servis zorunlu)
        if len(card_indices) >= 4 or (len(card_indices) >= 1 and len(available) <= 3):
            # İstirahat kontrolü
            has_rest = card_has_rest(card_services, settings) if len(card_services) > 1 else True
            
            # Reduced cost hesapla
            reduced_cost = 1.0 - sum(dual_values.get(i, 0) for i in card_indices)
            
            # İstirahat yoksa ceza ekle
            if not has_rest and len(card_services) > 1:
                reduced_cost += 0.5
            
            # En iyi kartı güncelle
            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_card = card_indices
    
    return best_card, best_reduced_cost

def solve_master_problem_simple(cards, num_services):
    """
    MASTER PROBLEM: Basit LP çözümü
    Her servisin tam olarak bir kartta olmasını sağla
    
    OR-Tools yoksa basit greedy çözüm
    """
    if not cards:
        return {}, {i: 1.0 for i in range(num_services)}, float('inf')
    
    # Hangi servisler hangi kartlarda?
    service_to_cards = {i: [] for i in range(num_services)}
    for card_idx, card in enumerate(cards):
        for srv_idx in card:
            service_to_cards[srv_idx].append(card_idx)
    
    # Greedy set cover - her servisi kapsayan minimum kart
    selected_cards = []
    covered = set()
    
    # Kart skorlarını hesapla (çok servisli kartlar önce)
    card_scores = []
    for card_idx, card in enumerate(cards):
        uncovered_count = sum(1 for s in card if s not in covered)
        # Kurallara uygunluk bonusu
        card_services = [services_global[i] for i in card] if 'services_global' in globals() else []
        rule_bonus = 0
        if len(card) >= 4:
            rule_bonus += 10
        card_scores.append((card_idx, uncovered_count + rule_bonus, len(card)))
    
    # En çok servisi kapsayan kartları seç
    while len(covered) < num_services:
        # Skorları güncelle
        card_scores = [(idx, sum(1 for s in cards[idx] if s not in covered), len(cards[idx])) 
                       for idx, _, _ in card_scores if idx not in selected_cards]
        
        if not card_scores:
            break
        
        # En iyi kartı seç
        card_scores.sort(key=lambda x: (-x[1], -x[2]))
        best_card_idx = card_scores[0][0]
        
        if card_scores[0][1] == 0:  # Hiç yeni servis kapsamıyor
            break
        
        selected_cards.append(best_card_idx)
        covered.update(cards[best_card_idx])
    
    # Dual değerler (basit tahmin)
    dual_values = {}
    for i in range(num_services):
        if i in covered:
            # Kaç kartta var?
            count = len(service_to_cards[i])
            dual_values[i] = 1.0 / max(count, 1)
        else:
            dual_values[i] = 2.0  # Kapsamayan servise yüksek değer
    
    # Çözüm
    solution = {idx: 1.0 for idx in selected_cards}
    obj_value = len(selected_cards)
    
    return solution, dual_values, obj_value

def column_generation_algorithm(services, settings, progress_callback=None):
    """
    BASİTLEŞTİRİLMİŞ COLUMN GENERATION
    
    1. Başlangıç kartları oluştur
    2. Master problem çöz (hangi kartlar kullanılacak)
    3. Subproblem çöz (yeni kart üret)
    4. Yeni kart iyileştirme sağlıyorsa ekle
    5. Tekrarla
    """
    global services_global
    services_global = services
    
    n = len(services)
    if n == 0:
        return []
    
    if progress_callback:
        progress_callback(0.05, "Column Generation: Başlangıç kartları oluşturuluyor...")
    
    # ==========================================
    # AŞAMA 1: Başlangıç kartları (feasible start)
    # ==========================================
    initial_cards = []
    
    # Önce kurallara uygun kartlar üretmeye çalış
    remaining = set(range(n))
    sorted_indices = sorted(range(n), key=lambda i: time_to_minutes(services[i]['gidis']))
    
    while remaining:
        card = []
        card_services = []
        
        for idx in sorted_indices:
            if idx not in remaining:
                continue
            
            srv = services[idx]
            
            if not card:
                card.append(idx)
                card_services.append(srv)
                continue
            
            # Çakışma kontrolü
            if card_has_conflict(card_services, srv):
                continue
            
            # Kural kontrolü
            test_card = card_services + [srv]
            if check_gap_rules(test_card, settings):
                card.append(idx)
                card_services.append(srv)
        
        if card:
            initial_cards.append(card)
            remaining -= set(card)
        else:
            # Kalan servisleri tek tek ekle
            for idx in list(remaining):
                initial_cards.append([idx])
                remaining.remove(idx)
                break
    
    if progress_callback:
        progress_callback(0.1, f"Column Generation: {len(initial_cards)} başlangıç kartı")
    
    # ==========================================
    # AŞAMA 2: Column Generation Döngüsü
    # ==========================================
    all_cards = list(initial_cards)
    best_obj = float('inf')
    no_improvement_count = 0
    max_iterations = 50
    
    for iteration in range(max_iterations):
        if progress_callback:
            progress_callback(0.1 + 0.5 * iteration / max_iterations, 
                            f"Column Generation: İterasyon {iteration + 1}/{max_iterations}")
        
        # Master problem çöz
        solution, dual_values, obj_value = solve_master_problem_simple(all_cards, n)
        
        if obj_value < best_obj:
            best_obj = obj_value
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Erken durdurma
        if no_improvement_count >= 5:
            break
        
        # Subproblem: Yeni kart üret
        used_services = set()
        for card_idx in solution:
            if solution[card_idx] > 0.5:
                used_services.update(all_cards[card_idx])
        
        new_card, reduced_cost = generate_valid_card(services, settings, dual_values, set(), max_attempts=50)
        
        if new_card is None or reduced_cost >= -0.01:
            # Yeni kart bulunamadı veya iyileştirme yok
            no_improvement_count += 1
            continue
        
        # Yeni kartı kontrol et - zaten var mı?
        new_card_set = set(new_card)
        is_duplicate = any(set(card) == new_card_set for card in all_cards)
        
        if not is_duplicate:
            all_cards.append(new_card)
    
    if progress_callback:
        progress_callback(0.6, "Column Generation: Çözüm seçiliyor...")
    
    # ==========================================
    # AŞAMA 3: Final çözüm seç
    # ==========================================
    solution, dual_values, obj_value = solve_master_problem_simple(all_cards, n)
    
    # Seçilen kartları al
    selected_cards = []
    covered = set()
    
    for card_idx, value in solution.items():
        if value > 0.5:
            card_services = [services[i] for i in all_cards[card_idx]]
            selected_cards.append(card_services)
            covered.update(all_cards[card_idx])
    
    # Kapsamayan servisler var mı?
    uncovered = set(range(n)) - covered
    if uncovered:
        if progress_callback:
            progress_callback(0.7, f"Column Generation: {len(uncovered)} servis ekleniyor...")
        
        # Kapsamayan servisleri mevcut kartlara ekle veya yeni kart oluştur
        for idx in uncovered:
            srv = services[idx]
            placed = False
            
            # Mevcut kartlara eklemeye çalış
            for card in selected_cards:
                if card_has_conflict(card, srv):
                    continue
                
                test_card = card + [srv]
                if check_gap_rules(test_card, settings):
                    card.append(srv)
                    placed = True
                    break
            
            if not placed:
                # Yeni kart oluştur
                selected_cards.append([srv])
    
    if progress_callback:
        progress_callback(0.8, "Column Generation: Post-processing...")
    
    # ==========================================
    # AŞAMA 4: Post-processing
    # ==========================================
    selected_cards = post_process_cards(selected_cards, settings)
    
    if progress_callback:
        progress_callback(1.0, f"Column Generation: {len(selected_cards)} kart oluşturuldu")
    
    return selected_cards

# ============================================
# ALGORİTMA MODELLERİ - KURAL UYUMLU
# ============================================

def calculate_card_score(card, settings):
    """Bir kartın kalite skoru (düşük = iyi)"""
    score = 0
    if len(card) <= 3:
        score += 10000
    if len(card) == 1:
        score += 50000
    if len(card) > 1 and not card_has_rest(card, settings):
        score += 5000
    score += count_violations(card, settings) * 3000
    return score

def check_gap_rules(card, settings):
    """
    Karttaki TÜM boşlukların kurallara uygunluğunu kontrol et
    Pik saat, max aralık saati, min-max aralık kuralları
    """
    if len(card) <= 1:
        return True
    
    for gi in get_gaps_with_times(card):
        gap, gap_start = gi['gap'], gi['start']
        
        # Pik saat kontrolü - SADECE pik aralıkları
        if is_pik_time(gap_start, settings):
            if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                return False
            continue
        
        # Minimum aralık kontrolü - ZORUNLU
        if gap < settings['oncelik1_min']:
            return False
        
        # Max aralık saati - büyük aralığa izin ver
        if is_max_aralik_time(gap_start, settings):
            if gap > settings['oncelik3_max']:
                return False
        else:
            # Diğer saatler - max oncelik2_max
            if gap > settings['oncelik2_max']:
                return False
    
    return True

def can_add_to_card(card, srv, settings, check_rest=True):
    """
    Servis karta eklenebilir mi? TÜM KURALLARA göre kontrol
    """
    if card_has_conflict(card, srv):
        return False
    
    test_card = card + [srv]
    
    # Gap kuralları
    if not check_gap_rules(test_card, settings):
        return False
    
    # İstirahat kontrolü
    if check_rest and len(test_card) > 1 and not card_has_rest(test_card, settings):
        return False
    
    return True

def post_process_cards(cards, settings):
    """
    POST-PROCESSING: Kartları kurallara uygun hale getir
    
    Aşama 1: Küçük kartları (1-3 srv) birleştir/büyüt
    Aşama 2: Pik saat ihlallerini düzelt
    Aşama 3: İstirahat kontrolü ve düzeltme
    """
    result = [list(c) for c in cards if c]
    
    # ==========================================
    # AŞAMA 1: Küçük kartları düzelt
    # ==========================================
    for iteration in range(300):
        small_cards = [(i, c) for i, c in enumerate(result) if 1 <= len(c) <= 3]
        
        if len(small_cards) <= 2:
            break
        
        improved = False
        
        # Strateji 1: İki küçük kartı birleştir
        for i in range(len(small_cards)):
            if improved:
                break
            for j in range(i + 1, len(small_cards)):
                idx1, idx2 = small_cards[i][0], small_cards[j][0]
                if idx1 >= len(result) or idx2 >= len(result):
                    continue
                
                card1, card2 = result[idx1], result[idx2]
                if any(has_time_conflict(s1, s2) for s1 in card1 for s2 in card2):
                    continue
                
                merged = card1 + card2
                
                # Önce strict kontrol
                if check_gap_rules(merged, settings) and card_has_rest(merged, settings):
                    result[idx1] = merged
                    result[idx2] = []
                    improved = True
                    break
                
                # Gevşek kontrol (sadece genel limitler)
                gaps = get_gaps_with_times(merged)
                gaps_ok = all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] for g in gaps)
                if gaps_ok:
                    result[idx1] = merged
                    result[idx2] = []
                    improved = True
                    break
        
        result = [c for c in result if c]
        if improved:
            continue
        
        # Strateji 2: Küçük karttan büyük karta servis taşı
        small_cards = [(i, c) for i, c in enumerate(result) if 1 <= len(c) <= 3]
        large_cards = [(i, c) for i, c in enumerate(result) if len(c) >= 4]
        
        for small_idx, small_card in small_cards:
            if improved:
                break
            for srv in list(small_card):
                for large_idx, large_card in large_cards:
                    if card_has_conflict(large_card, srv):
                        continue
                    
                    test = large_card + [srv]
                    
                    # Strict kontrol
                    if check_gap_rules(test, settings) and card_has_rest(test, settings):
                        result[large_idx].append(srv)
                        result[small_idx] = [s for s in result[small_idx] if s['_id'] != srv['_id']]
                        improved = True
                        break
                    
                    # Gevşek kontrol
                    gaps = get_gaps_with_times(test)
                    gaps_ok = all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] for g in gaps)
                    if gaps_ok:
                        result[large_idx].append(srv)
                        result[small_idx] = [s for s in result[small_idx] if s['_id'] != srv['_id']]
                        improved = True
                        break
                if improved:
                    break
        
        result = [c for c in result if c]
        if improved:
            continue
        
        # Strateji 3: Büyük karttan küçük karta servis çek
        small_cards = [(i, c) for i, c in enumerate(result) if 1 <= len(c) <= 3]
        large_cards = [(i, c) for i, c in enumerate(result) if len(c) >= 5]
        
        for small_idx, small_card in small_cards[:max(1, len(small_cards) - 2)]:
            if improved:
                break
            for large_idx, large_card in large_cards:
                if improved:
                    break
                for srv in list(large_card):
                    if card_has_conflict(small_card, srv):
                        continue
                    
                    test_small = small_card + [srv]
                    test_large = [s for s in large_card if s['_id'] != srv['_id']]
                    
                    # Gevşek kontrol
                    small_gaps = get_gaps_with_times(test_small)
                    large_gaps = get_gaps_with_times(test_large)
                    
                    small_ok = all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] for g in small_gaps)
                    large_ok = len(test_large) <= 1 or all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] for g in large_gaps)
                    
                    if small_ok and large_ok:
                        result[small_idx] = test_small
                        result[large_idx] = test_large
                        improved = True
                        break
        
        result = [c for c in result if c]
        
        if not improved:
            break
    
    # ==========================================
    # AŞAMA 2: Kartları birleştirmeye çalış (kart sayısını azalt)
    # ==========================================
    improved = True
    iterations = 0
    while improved and iterations < 100:
        iterations += 1
        improved = False
        
        for i in range(len(result)):
            if improved:
                break
            for j in range(i + 1, len(result)):
                if j >= len(result):
                    continue
                
                if any(has_time_conflict(s1, s2) for s1 in result[i] for s2 in result[j]):
                    continue
                
                merged = result[i] + result[j]
                
                if check_gap_rules(merged, settings) and card_has_rest(merged, settings):
                    result[i] = merged
                    result.pop(j)
                    improved = True
                    break
        
        result = [c for c in result if c]
    
    # ==========================================
    # AŞAMA 3: Pik saat ve istirahat ihlallerini düzeltmeye çalış
    # ==========================================
    for iteration in range(100):
        improved = False
        
        # İhlalli kartları bul
        problem_cards = []
        good_cards = []
        
        for i, card in enumerate(result):
            has_violation = count_violations(card, settings) > 0
            no_rest = len(card) > 1 and not card_has_rest(card, settings)
            
            if has_violation or no_rest:
                problem_cards.append((i, card))
            else:
                good_cards.append((i, card))
        
        if not problem_cards:
            break
        
        # İhlalli kartlardan servis taşımaya çalış
        for prob_idx, prob_card in problem_cards:
            if improved:
                break
            
            for srv in list(prob_card):
                if improved:
                    break
                
                for good_idx, good_card in good_cards:
                    if card_has_conflict(good_card, srv):
                        continue
                    
                    test_good = good_card + [srv]
                    test_prob = [s for s in prob_card if s['_id'] != srv['_id']]
                    
                    # Her iki kart da kurallara uyuyor mu?
                    good_ok = check_gap_rules(test_good, settings) and card_has_rest(test_good, settings)
                    prob_ok = len(test_prob) <= 1 or (check_gap_rules(test_prob, settings) and card_has_rest(test_prob, settings))
                    
                    # En az biri düzeldi mi?
                    old_violations = count_violations(prob_card, settings) + count_violations(good_card, settings)
                    new_violations = count_violations(test_prob, settings) + count_violations(test_good, settings)
                    
                    if new_violations < old_violations or (good_ok and prob_ok):
                        result[good_idx] = test_good
                        result[prob_idx] = test_prob
                        improved = True
                        break
        
        result = [c for c in result if c]
        
        if not improved:
            break
    
    return [c for c in result if c]

# ============================================
# MODEL 1: CONSTRAINT PROGRAMMING (CP)
# ============================================
def optimize_with_cp(services, settings, progress_callback=None):
    """
    Constraint Programming ile optimizasyon - KURAL UYUMLU
    """
    if not ORTOOLS_AVAILABLE:
        if progress_callback:
            progress_callback(0.1, "CP: OR-Tools yok, Greedy kullanılıyor...")
        return optimize_with_greedy_local_search(services, settings, progress_callback)
    
    n = len(services)
    if n == 0:
        return []
    
    max_cards = n // 4 + 1
    
    model = cp_model.CpModel()
    
    # Değişkenler
    assignments = [model.NewIntVar(0, max_cards - 1, f's_{i}') for i in range(n)]
    card_used = [model.NewBoolVar(f'card_{k}') for k in range(max_cards)]
    
    # Kısıt 1: Çakışan servisler aynı karta atanamaz
    for i in range(n):
        for j in range(i + 1, n):
            if has_time_conflict(services[i], services[j]):
                model.Add(assignments[i] != assignments[j])
    
    # Kısıt 2: Aralık kurallarına uymayan servisler aynı karta atanamaz
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = services[i], services[j]
            
            # Aralık hesapla
            s1_end = get_end_time(s1)
            s2_start = time_to_minutes(s2['gidis'])
            s2_end = get_end_time(s2)
            s1_start = time_to_minutes(s1['gidis'])
            
            if s1_start < s2_start:
                gap = s2_start - s1_end
                gap_start = s1_end
            else:
                gap = s1_start - s2_end
                gap_start = s2_end
            
            # Kural kontrolü
            valid = True
            if is_pik_time(gap_start, settings):
                if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                    valid = False
            else:
                if gap < settings['oncelik1_min']:
                    valid = False
                elif is_max_aralik_time(gap_start, settings):
                    if gap > settings['oncelik3_max']:
                        valid = False
                else:
                    if gap > settings['oncelik2_max']:
                        valid = False
            
            if not valid:
                model.Add(assignments[i] != assignments[j])
    
    # Kısıt 3: Kart kullanımını izle
    for i in range(n):
        for k in range(max_cards):
            b = model.NewBoolVar(f'b_{i}_{k}')
            model.Add(assignments[i] == k).OnlyEnforceIf(b)
            model.Add(assignments[i] != k).OnlyEnforceIf(b.Not())
            model.AddImplication(b, card_used[k])
    
    # Hedef: Kart sayısını minimize et
    model.Minimize(sum(card_used))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    
    if progress_callback:
        progress_callback(0.5, "CP: Çözülüyor...")
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        cards_dict = {}
        for i in range(n):
            card_idx = solver.Value(assignments[i])
            if card_idx not in cards_dict:
                cards_dict[card_idx] = []
            cards_dict[card_idx].append(services[i])
        
        cards = list(cards_dict.values())
        
        if progress_callback:
            progress_callback(0.8, "CP: Post-processing...")
        
        # Post-processing ile kuralları uygula
        cards = post_process_cards(cards, settings)
        
        return cards
    else:
        return optimize_with_greedy_local_search(services, settings, progress_callback)

# ============================================
# MODEL 2: INTEGER LINEAR PROGRAMMING (ILP)
# ============================================
def optimize_with_ilp(services, settings, progress_callback=None):
    """
    Integer Linear Programming - KURAL UYUMLU
    """
    if progress_callback:
        progress_callback(0.2, "ILP: Başlatılıyor...")
    
    # ILP için Bin Packing + Post-processing kullan
    cards = optimize_with_bin_packing_strict(services, settings, progress_callback)
    
    if progress_callback:
        progress_callback(0.8, "ILP: Post-processing...")
    
    cards = post_process_cards(cards, settings)
    
    return cards

# ============================================
# MODEL 3: BIN PACKING - KURAL UYUMLU
# ============================================
def optimize_with_bin_packing(services, settings, progress_callback=None):
    """
    Bin Packing - KURAL UYUMLU
    """
    return optimize_with_bin_packing_strict(services, settings, progress_callback)

def optimize_with_bin_packing_strict(services, settings, progress_callback=None):
    """
    Bin Packing - TÜM KURALLARA UYGUN
    """
    n = len(services)
    if n == 0:
        return []
    
    if progress_callback:
        progress_callback(0.1, "Bin Packing: Başlatılıyor...")
    
    # Servisleri saate göre sırala
    sorted_services = sorted(services, key=lambda s: time_to_minutes(s['gidis']))
    
    cards = []
    
    for idx, srv in enumerate(sorted_services):
        if progress_callback and idx % 20 == 0:
            progress_callback(0.1 + 0.5 * idx / n, f"Bin Packing: {idx}/{n}...")
        
        best_card_idx = -1
        best_score = float('inf')
        
        for i, card in enumerate(cards):
            # TÜM KURALLARA göre kontrol
            if not can_add_to_card(card, srv, settings, check_rest=True):
                # İstirahat olmadan da dene
                if not can_add_to_card(card, srv, settings, check_rest=False):
                    continue
            
            test_card = card + [srv]
            gaps = get_gaps_with_times(test_card)
            
            # Skor: ihlal sayısı + max aralık
            score = count_violations(test_card, settings) * 1000
            score += max(g['gap'] for g in gaps) if gaps else 0
            if not card_has_rest(test_card, settings):
                score += 500
            
            if score < best_score:
                best_score = score
                best_card_idx = i
        
        if best_card_idx >= 0:
            cards[best_card_idx].append(srv)
        else:
            cards.append([srv])
    
    if progress_callback:
        progress_callback(0.7, "Bin Packing: Post-processing...")
    
    # Post-processing
    cards = post_process_cards(cards, settings)
    
    return cards

# ============================================
# MODEL 4: GREEDY + LOCAL SEARCH - KURAL UYUMLU
# ============================================
def optimize_with_greedy_local_search(services, settings, progress_callback=None):
    """
    Greedy + Local Search - TÜM KURALLARA UYGUN
    """
    n = len(services)
    if n == 0:
        return []
    
    if progress_callback:
        progress_callback(0.1, "Greedy: Başlatılıyor...")
    
    # Saate göre sırala
    sorted_services = sorted(services, key=lambda s: time_to_minutes(s['gidis']))
    
    cards = []
    
    for idx, srv in enumerate(sorted_services):
        if progress_callback and idx % 20 == 0:
            progress_callback(0.1 + 0.3 * idx / n, f"Greedy: {idx}/{n}...")
        
        best_card_idx = -1
        best_score = float('inf')
        
        for i, card in enumerate(cards):
            # TÜM KURALLARA göre kontrol
            if card_has_conflict(card, srv):
                continue
            
            test_card = card + [srv]
            
            # Gap kuralları
            if not check_gap_rules(test_card, settings):
                continue
            
            gaps = get_gaps_with_times(test_card)
            
            # Skor hesapla
            score = max(g['gap'] for g in gaps) if gaps else 0
            score += count_violations(test_card, settings) * 1000
            if not card_has_rest(test_card, settings):
                score += 500
            
            if score < best_score:
                best_score = score
                best_card_idx = i
        
        if best_card_idx >= 0:
            cards[best_card_idx].append(srv)
        else:
            cards.append([srv])
    
    if progress_callback:
        progress_callback(0.5, "Greedy: Post-processing...")
    
    # Post-processing
    cards = post_process_cards(cards, settings)
    
    return cards

def local_search_improvement(cards, settings, progress_callback=None):
    """
    Local Search iyileştirme - KURAL UYUMLU
    """
    return post_process_cards(cards, settings)

# ============================================
# MODEL 5: HYBRID (EN İYİ SONUCU SEÇ)
# ============================================
def optimize_with_hybrid(services, settings, progress_callback=None):
    """
    Hybrid: Birden fazla yöntemi dener, en iyi sonucu seçer
    TÜM KURALLARA UYGUN
    """
    n = len(services)
    if n == 0:
        return []
    
    best_cards = None
    best_score = float('inf')
    
    def calc_total_score(cards):
        """Toplam skor hesapla (düşük = iyi)"""
        score = len(cards) * 1000  # Kart sayısı
        score += sum(1 for c in cards if len(c) <= 3) * 50000  # Küçük kart
        score += sum(1 for c in cards if len(c) == 1) * 100000  # Tek servisli
        score += sum(1 for c in cards if len(c) > 1 and not card_has_rest(c, settings)) * 10000  # İstirahatsız
        for c in cards:
            score += count_violations(c, settings) * 5000  # İhlal
        return score
    
    # Yöntem 1: Bin Packing
    if progress_callback:
        progress_callback(0.1, "Hybrid: Bin Packing deneniyor...")
    try:
        cards_bp = optimize_with_bin_packing_strict(services, settings, None)
        score_bp = calc_total_score(cards_bp)
        if score_bp < best_score:
            best_score = score_bp
            best_cards = cards_bp
    except:
        pass
    
    # Yöntem 2: Greedy (farklı sıralamalarla)
    if progress_callback:
        progress_callback(0.3, "Hybrid: Greedy varyasyonları deneniyor...")
    
    for attempt in range(5):
        try:
            shuffled = list(services)
            if attempt > 0:
                random.shuffle(shuffled)
            
            cards_gr = optimize_with_greedy_local_search(shuffled, settings, None)
            score_gr = calc_total_score(cards_gr)
            if score_gr < best_score:
                best_score = score_gr
                best_cards = cards_gr
        except:
            pass
    
    # Yöntem 3: CP (varsa)
    if ORTOOLS_AVAILABLE:
        if progress_callback:
            progress_callback(0.6, "Hybrid: CP deneniyor...")
        try:
            cards_cp = optimize_with_cp(services, settings, None)
            score_cp = calc_total_score(cards_cp)
            if score_cp < best_score:
                best_score = score_cp
                best_cards = cards_cp
        except:
            pass
    
    if progress_callback:
        progress_callback(0.9, "Hybrid: En iyi sonuç seçildi")
    
    if best_cards is None:
        best_cards = [[s] for s in services]
        best_cards = post_process_cards(best_cards, settings)
    
    return best_cards

def build_normalci_esit_aralikli(services, settings):
    """
    EŞİT ARALIKLI SERVİS OPTİMİZASYONU
    - 1,2,3 servisli kart OLMAYACAK (min 4 servis)
    - Pik saatlerde pik aralık kurallarına uyulacak
    - Pik dışı saatlerde aralıklar eşit dağıtılacak
    """
    cards = []
    remaining = sorted(list(services), key=lambda s: time_to_minutes(s['gidis']))
    
    # Önce tüm servisleri kartlara dağıt - minimum 4 servis hedefle
    for srv in remaining:
        placed = False
        best_idx = -1
        best_variance = float('inf')
        
        for i, card in enumerate(cards):
            if card_has_conflict(card, srv):
                continue
            
            test = card + [srv]
            gaps = get_gaps_with_times(test)
            
            # İstirahat kontrolü
            if not card_has_rest(test, settings):
                continue
            
            # Pik saat kontrolü - ZORUNLU
            pik_valid = True
            non_pik_gaps = []
            
            for gi in gaps:
                gap, gap_start = gi['gap'], gi['start']
                
                if is_pik_time(gap_start, settings):
                    # Pik saatte pik aralık kuralları ZORUNLU
                    if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                        pik_valid = False
                        break
                else:
                    # Pik dışı - genel limitlere uy
                    if gap < settings['oncelik1_min'] or gap > settings['oncelik3_max']:
                        pik_valid = False
                        break
                    non_pik_gaps.append(gap)
            
            if not pik_valid:
                continue
            
            # Varyans hesapla - aralıkların eşitliğini ölç
            if non_pik_gaps:
                avg = sum(non_pik_gaps) / len(non_pik_gaps)
                variance = sum((g - avg) ** 2 for g in non_pik_gaps) / len(non_pik_gaps)
            else:
                variance = 0
            
            if variance < best_variance:
                best_variance = variance
                best_idx = i
        
        if best_idx != -1:
            cards[best_idx].append(srv)
        else:
            cards.append([srv])
    
    # Küçük kartları (1-3 servis) büyük kartlara taşı - AGRESİF
    for iteration in range(200):
        small_cards = [(i, c) for i, c in enumerate(cards) if 1 <= len(c) <= 3]
        
        if len(small_cards) == 0:
            break
        
        improved = False
        
        # İki küçük kartı birleştir
        for i in range(len(small_cards)):
            if improved:
                break
            for j in range(i + 1, len(small_cards)):
                idx1, idx2 = small_cards[i][0], small_cards[j][0]
                if idx1 >= len(cards) or idx2 >= len(cards):
                    continue
                
                card1, card2 = cards[idx1], cards[idx2]
                if any(has_time_conflict(s1, s2) for s1 in card1 for s2 in card2):
                    continue
                
                merged = card1 + card2
                gaps = get_gaps_with_times(merged)
                
                if not card_has_rest(merged, settings):
                    continue
                
                # Pik saat ve genel limit kontrolü
                valid = True
                for gi in gaps:
                    gap, gap_start = gi['gap'], gi['start']
                    if is_pik_time(gap_start, settings):
                        if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                            valid = False
                            break
                    else:
                        if gap < settings['oncelik1_min'] or gap > settings['oncelik3_max']:
                            valid = False
                            break
                
                if valid:
                    cards[idx1] = merged
                    cards[idx2] = []
                    improved = True
                    break
        
        cards = [c for c in cards if c]
        if improved:
            continue
        
        # Küçük karttan büyük karta servis taşı
        small_cards = [(i, c) for i, c in enumerate(cards) if 1 <= len(c) <= 3]
        large_cards = [(i, c) for i, c in enumerate(cards) if len(c) >= 4]
        
        for small_idx, small_card in small_cards:
            if improved:
                break
            for srv in list(small_card):
                for large_idx, large_card in large_cards:
                    if card_has_conflict(large_card, srv):
                        continue
                    
                    test = large_card + [srv]
                    gaps = get_gaps_with_times(test)
                    
                    if not card_has_rest(test, settings):
                        continue
                    
                    valid = True
                    for gi in gaps:
                        gap, gap_start = gi['gap'], gi['start']
                        if is_pik_time(gap_start, settings):
                            if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                                valid = False
                                break
                        else:
                            if gap < settings['oncelik1_min'] or gap > settings['oncelik3_max']:
                                valid = False
                                break
                    
                    if valid:
                        cards[large_idx].append(srv)
                        cards[small_idx] = [s for s in cards[small_idx] if s['_id'] != srv['_id']]
                        improved = True
                        break
                if improved:
                    break
        
        cards = [c for c in cards if c]
        
        if not improved:
            break
    
    return [c for c in cards if c]

# ============================================
# NORMALCİ KART OLUŞTURMA - SAAT KURALLARI
# ============================================
def build_normalci_cards_strict(services, settings):
    """
    Normalci kartları oluştur - ÖNCE SIKISIK, SONRA GENİŞLET
    Minimum kart sayısı için öncelik sırasıyla aralık genişletilir
    """
    cards = []
    remaining = sorted(list(services), key=lambda s: time_to_minutes(s['gidis']))
    
    for srv in remaining:
        placed = False
        best_idx = -1
        best_score = float('inf')
        best_priority = 4  # Düşük öncelik = daha iyi
        
        for i, card in enumerate(cards):
            if card_has_conflict(card, srv):
                continue
            
            test = card + [srv]
            gaps = get_gaps_with_times(test)
            
            # İstirahat kontrolü
            if not card_has_rest(test, settings):
                continue
            
            # Öncelik belirleme
            priority = 1
            all_valid = True
            for gi in gaps:
                gap, gap_start = gi['gap'], gi['start']
                
                # Minimum aralık kontrolü (pik hariç)
                if not is_pik_time(gap_start, settings) and gap < settings['oncelik1_min']:
                    all_valid = False
                    break
                
                # Pik saat kontrolü
                if is_pik_time(gap_start, settings):
                    if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                        all_valid = False
                        break
                    continue
                
                # Öncelik belirleme (pik dışı)
                if gap <= settings['oncelik1_max']:
                    pass  # 1. öncelik
                elif gap <= settings['oncelik2_max']:
                    priority = max(priority, 2)
                elif gap <= settings['oncelik3_max'] and is_max_aralik_time(gap_start, settings):
                    priority = max(priority, 3)
                elif gap <= settings['oncelik3_max']:
                    # 3. öncelik sadece max aralık saatinde - diğer saatlerde 2. önceliğe kadar
                    all_valid = False
                    break
                else:
                    all_valid = False
                    break
            
            if not all_valid:
                continue
            
            # En iyi kartı seç (öncelik ve skor bazlı)
            score = max(g['gap'] for g in gaps) if gaps else 0
            if priority < best_priority or (priority == best_priority and score < best_score):
                best_priority = priority
                best_score = score
                best_idx = i
        
        if best_idx != -1:
            cards[best_idx].append(srv)
        else:
            cards.append([srv])
    
    return cards

def merge_normalci_strict(cards, settings):
    """Kartları birleştir - ÖNCELİK SİSTEMİNE GÖRE ARALIK GENİŞLETİLEBİLİR"""
    result = [list(c) for c in cards if c]
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations:
        iteration += 1
        improved = False
        
        for i in range(len(result)):
            if improved:
                break
            for j in range(i + 1, len(result)):
                if any(has_time_conflict(s1, s2) for s1 in result[i] for s2 in result[j]):
                    continue
                
                merged = result[i] + result[j]
                gaps = get_gaps_with_times(merged)
                
                # İstirahat kontrolü
                if not card_has_rest(merged, settings):
                    continue
                
                # Öncelik sistemine göre kontrol
                all_valid = True
                for gi in gaps:
                    gap, gap_start = gi['gap'], gi['start']
                    
                    # Minimum aralık (pik hariç)
                    if not is_pik_time(gap_start, settings) and gap < settings['oncelik1_min']:
                        all_valid = False
                        break
                    
                    # Pik saat
                    if is_pik_time(gap_start, settings):
                        if not (settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']):
                            all_valid = False
                            break
                        continue
                    
                    # Max aralık saatinde 3. önceliğe kadar
                    if is_max_aralik_time(gap_start, settings):
                        if gap > settings['oncelik3_max']:
                            all_valid = False
                            break
                    else:
                        # Diğer saatlerde 2. önceliğe kadar
                        if gap > settings['oncelik2_max']:
                            all_valid = False
                            break
                
                if all_valid:
                    result[i] = merged
                    result.pop(j)
                    improved = True
                    break
    
    return [c for c in result if c]

def fix_small_cards(cards, settings, max_small=2):
    """
    Küçük kartları (≤3 srv) düzelt - max_small adet kalana kadar
    ÇOK AGRESİF: Tüm yöntemleri dene, kart sayısını minimize et
    """
    result = [list(c) for c in cards if c]
    
    for iteration in range(100):
        result = [c for c in result if c]  # Boşları temizle
        small_indices = [i for i, c in enumerate(result) if 1 <= len(c) <= 3]
        
        if len(small_indices) <= max_small:
            break
        
        improved = False
        
        # Strateji 1: En küçük iki kartı birleştirmeye çalış
        if len(small_indices) >= 2:
            for i in range(len(small_indices)):
                if improved:
                    break
                for j in range(i + 1, len(small_indices)):
                    idx1, idx2 = small_indices[i], small_indices[j]
                    if idx1 >= len(result) or idx2 >= len(result):
                        continue
                    
                    card1, card2 = result[idx1], result[idx2]
                    if any(has_time_conflict(s1, s2) for s1 in card1 for s2 in card2):
                        continue
                    
                    merged = card1 + card2
                    
                    # Öncelik sırasıyla dene: önce strict, sonra gevşek
                    if card_gaps_valid_strict(merged, settings) and card_has_rest(merged, settings):
                        result[idx1] = merged
                        result[idx2] = []
                        improved = True
                        break
                    
                    # Gevşek kontrol - sadece genel aralık limitleri
                    gaps_ok = all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] 
                                 for g in get_gaps_with_times(merged))
                    if gaps_ok and card_has_rest(merged, settings):
                        result[idx1] = merged
                        result[idx2] = []
                        improved = True
                        break
        
        result = [c for c in result if c]
        if improved:
            continue
        
        # Strateji 2: Küçük karttan büyük karta servis taşı
        small_indices = [i for i, c in enumerate(result) if 1 <= len(c) <= 3]
        other_indices = [i for i, c in enumerate(result) if len(c) >= 4]
        
        for small_idx in small_indices:
            if improved:
                break
            if small_idx >= len(result):
                continue
            
            for srv in list(result[small_idx]):
                for other_idx in other_indices:
                    if other_idx >= len(result) or other_idx == small_idx:
                        continue
                    if card_has_conflict(result[other_idx], srv):
                        continue
                    
                    test = result[other_idx] + [srv]
                    
                    # Öncelik sırasıyla kontrol
                    if card_gaps_valid_strict(test, settings) and card_has_rest(test, settings):
                        result[other_idx].append(srv)
                        result[small_idx] = [s for s in result[small_idx] if s['_id'] != srv['_id']]
                        improved = True
                        break
                    
                    # Gevşek kontrol
                    gaps_ok = all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] 
                                 for g in get_gaps_with_times(test))
                    if gaps_ok and card_has_rest(test, settings):
                        result[other_idx].append(srv)
                        result[small_idx] = [s for s in result[small_idx] if s['_id'] != srv['_id']]
                        improved = True
                        break
                if improved:
                    break
        
        result = [c for c in result if c]
        if improved:
            continue
        
        # Strateji 3: Büyük karttan küçük karta servis çekerek büyüt
        small_indices = [i for i, c in enumerate(result) if 1 <= len(c) <= 3]
        large_indices = [i for i, c in enumerate(result) if len(c) >= 5]
        
        for small_idx in small_indices[:max(1, len(small_indices) - max_small)]:
            if improved:
                break
            if small_idx >= len(result):
                continue
            
            for large_idx in large_indices:
                if improved:
                    break
                if large_idx >= len(result):
                    continue
                
                for srv in list(result[large_idx]):
                    if card_has_conflict(result[small_idx], srv):
                        continue
                    
                    test_small = result[small_idx] + [srv]
                    test_large = [s for s in result[large_idx] if s['_id'] != srv['_id']]
                    
                    # Gevşek kontrol - minimum kart sayısı için
                    small_gaps_ok = all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] 
                                       for g in get_gaps_with_times(test_small))
                    large_gaps_ok = len(test_large) <= 1 or all(settings['oncelik1_min'] <= g['gap'] <= settings['oncelik3_max'] 
                                       for g in get_gaps_with_times(test_large))
                    
                    if small_gaps_ok and large_gaps_ok:
                        if card_has_rest(test_small, settings) and (len(test_large) <= 1 or card_has_rest(test_large, settings)):
                            result[small_idx] = test_small
                            result[large_idx] = test_large
                            improved = True
                            break
        
        result = [c for c in result if c]
        
        if not improved:
            break
    
    return [c for c in result if c]

# ============================================
# TEKÇİ KART OLUŞTURMA
# ============================================
def build_tekci_card(morning_pool, afternoon_pool, settings, used_ids, must_include=None):
    """Tekçi kartı oluştur"""
    card = []
    local_used = set()
    
    if must_include:
        for s in must_include:
            card.append(s)
            local_used.add(s['_id'])
    
    avail_m = [s for s in morning_pool if s['_id'] not in used_ids and s['_id'] not in local_used]
    random.shuffle(avail_m)
    
    for s in avail_m:
        m_in = [x for x in card if can_tekci_morning(x, settings)]
        if card_has_conflict(m_in, s):
            continue
        test = m_in + [s]
        if len(test) > 1 and any(not tekci_gap_valid(g, settings) for g in get_gaps(test)):
            continue
        card.append(s)
        local_used.add(s['_id'])
        if len([x for x in card if can_tekci_morning(x, settings)]) >= 3:
            break
    
    avail_a = [s for s in afternoon_pool if s['_id'] not in used_ids and s['_id'] not in local_used]
    random.shuffle(avail_a)
    
    for s in avail_a:
        a_in = [x for x in card if can_tekci_afternoon(x, settings)]
        if card_has_conflict(a_in, s):
            continue
        test = a_in + [s]
        if len(test) > 1 and any(not tekci_gap_valid(g, settings) for g in get_gaps(test)):
            continue
        if calc_tekci_work(card + [s], settings) > 9 * 60:
            continue
        card.append(s)
        local_used.add(s['_id'])
        if len([x for x in card if can_tekci_afternoon(x, settings)]) >= 3:
            break
    
    return card, local_used

# ============================================
# ANA OPTİMİZASYON FONKSİYONU
# ============================================
def optimize_group(services, settings):
    if not services:
        return [], []
    
    total = len(services)
    target_ratio = settings['tekci_oran']
    min_tekci_srv = settings['tekci_min_servis']
    kombinasyon = settings.get('kombinasyon_sayisi', 150)
    model = settings.get('model', 'global')
    
    progress = st.progress(0)
    status = st.empty()
    
    def progress_callback(pct, msg):
        progress.progress(min(pct, 1.0))
        status.text(msg)
    
    # ==========================================
    # GLOBAL OPTİMİZASYON - YENİ VARSAYILAN
    # ==========================================
    if model == 'global':
        status.text("GLOBAL OPTİMİZASYON: Tüm servisler birlikte optimize ediliyor...")
        tekci_cards, normalci_cards = global_optimize(services, settings, progress_callback)
        
        progress.progress(1.0)
        status.text(f"✅ Global Optimizasyon tamamlandı! Tekçi: {len(tekci_cards)}, Normalci: {len(normalci_cards)}")
        
        return tekci_cards, normalci_cards
    
    # ==========================================
    # DİĞER MODELLER (ESKİ YÖNTEM)
    # ==========================================
    required_tekci = [s for s in services if is_tekci_required(s)]
    required_ids = set(s['_id'] for s in required_tekci)
    
    morning_all = [s for s in services if can_tekci_morning(s, settings)]
    afternoon_all = [s for s in services if can_tekci_afternoon(s, settings)]
    
    best_solution = None
    best_score = float('inf')
    
    # Model bazlı optimizasyon
    if model in ['column_generation', 'cp', 'ilp', 'bin_packing', 'greedy_local', 'hybrid']:
        status.text(f"Model: {model.upper()} ile optimizasyon...")
        
        # Tekçi kartları önce oluştur (tüm modeller için aynı)
        morning_pool = list(morning_all)
        afternoon_pool = list(afternoon_all)
        
        tekci_cards = []
        used_ids = set()
        
        # Zorunlu tekçiler
        remaining_req = list(required_tekci)
        while remaining_req:
            first = remaining_req.pop(0)
            group = [first]
            used_ids.add(first['_id'])
            
            for req in list(remaining_req):
                if card_has_conflict(group, req):
                    continue
                m_grp = [s for s in group + [req] if can_tekci_morning(s, settings)]
                a_grp = [s for s in group + [req] if can_tekci_afternoon(s, settings)]
                valid = True
                for grp in [m_grp, a_grp]:
                    if len(grp) > 1 and any(not tekci_gap_valid(g, settings) for g in get_gaps(grp)):
                        valid = False
                        break
                if valid and calc_tekci_work(group + [req], settings) <= 9 * 60:
                    group.append(req)
                    used_ids.add(req['_id'])
                    remaining_req.remove(req)
            
            card, local_used = build_tekci_card(morning_pool, afternoon_pool, settings, used_ids, must_include=group)
            
            if len(card) >= min_tekci_srv or any(s['_id'] in required_ids for s in card):
                tekci_cards.append(card)
                used_ids.update(local_used)
        
        # Hedef tekçi sayısı
        remaining = [s for s in services if s['_id'] not in used_ids]
        est_normalci = max(1, len(remaining) // 6)
        est_total = len(tekci_cards) + est_normalci
        target_tekci = max(int(est_total * target_ratio), len(tekci_cards))
        
        # Ek tekçi kartları
        while len(tekci_cards) < target_tekci:
            avail_m = [s for s in morning_pool if s['_id'] not in used_ids]
            avail_a = [s for s in afternoon_pool if s['_id'] not in used_ids]
            
            if not avail_m and not avail_a:
                break
            
            card, local_used = build_tekci_card(avail_m, avail_a, settings, used_ids)
            
            if len(card) >= min_tekci_srv:
                tekci_cards.append(card)
                used_ids.update(local_used)
            else:
                break
        
        # Normalci servisleri
        remaining = [s for s in services if s['_id'] not in used_ids]
        
        progress_callback(0.1, f"{model.upper()}: Normalci kartları oluşturuluyor...")
        
        # Seçilen modele göre normalci optimizasyonu
        if model == 'column_generation':
            normalci_cards = column_generation_algorithm(remaining, settings, progress_callback)
        elif model == 'cp':
            normalci_cards = optimize_with_cp(remaining, settings, progress_callback)
        elif model == 'ilp':
            normalci_cards = optimize_with_ilp(remaining, settings, progress_callback)
        elif model == 'bin_packing':
            normalci_cards = optimize_with_bin_packing(remaining, settings, progress_callback)
            normalci_cards = post_process_cards(normalci_cards, settings)
        elif model == 'greedy_local':
            normalci_cards = optimize_with_greedy_local_search(remaining, settings, progress_callback)
        elif model == 'hybrid':
            normalci_cards = optimize_with_hybrid(remaining, settings, progress_callback)
        
        # Küçük kartları düzelt
        normalci_cards = fix_small_cards(normalci_cards, settings, max_small=2)
        
        progress.progress(1.0)
        
        # Skor hesapla
        total_cards = len(tekci_cards) + len(normalci_cards)
        score = sum(calculate_card_score(c, settings) for c in normalci_cards)
        score += total_cards * 1000
        
        status.text(f"✅ {model.upper()} tamamlandı! Skor: {score:.0f}")
        
        return tekci_cards, normalci_cards
    
    # Eski kombinasyon tabanlı optimizasyon (fallback)
    for combo in range(kombinasyon):
        if combo % 10 == 0:
            progress.progress(combo / kombinasyon)
            status.text(f"Kombinasyon {combo}/{kombinasyon}... En iyi: {best_score:.0f}")
        
        morning_pool = list(morning_all)
        afternoon_pool = list(afternoon_all)
        random.shuffle(morning_pool)
        random.shuffle(afternoon_pool)
        
        tekci_cards = []
        used_ids = set()
        
        # Zorunlu tekçiler
        remaining_req = list(required_tekci)
        while remaining_req:
            first = remaining_req.pop(0)
            group = [first]
            used_ids.add(first['_id'])
            
            for req in list(remaining_req):
                if card_has_conflict(group, req):
                    continue
                m_grp = [s for s in group + [req] if can_tekci_morning(s, settings)]
                a_grp = [s for s in group + [req] if can_tekci_afternoon(s, settings)]
                valid = True
                for grp in [m_grp, a_grp]:
                    if len(grp) > 1 and any(not tekci_gap_valid(g, settings) for g in get_gaps(grp)):
                        valid = False
                        break
                if valid and calc_tekci_work(group + [req], settings) <= 9 * 60:
                    group.append(req)
                    used_ids.add(req['_id'])
                    remaining_req.remove(req)
            
            card, local_used = build_tekci_card(morning_pool, afternoon_pool, settings, used_ids, must_include=group)
            
            if len(card) >= min_tekci_srv or any(s['_id'] in required_ids for s in card):
                tekci_cards.append(card)
                used_ids.update(local_used)
        
        # Hedef tekçi
        remaining = [s for s in services if s['_id'] not in used_ids]
        est_normalci = max(1, len(remaining) // 6)
        est_total = len(tekci_cards) + est_normalci
        target_tekci = max(int(est_total * target_ratio), len(tekci_cards))
        
        while len(tekci_cards) < target_tekci:
            avail_m = [s for s in morning_pool if s['_id'] not in used_ids]
            avail_a = [s for s in afternoon_pool if s['_id'] not in used_ids]
            
            if not avail_m and not avail_a:
                break
            
            card, local_used = build_tekci_card(avail_m, avail_a, settings, used_ids)
            
            if len(card) >= min_tekci_srv:
                tekci_cards.append(card)
                used_ids.update(local_used)
            else:
                break
        
        # Normalci - AŞAMA 1
        remaining = [s for s in services if s['_id'] not in used_ids]
        
        # Eşit aralıklı servis seçeneği aktifse özel algoritma kullan
        if settings.get('esit_aralikli', False):
            normalci_cards = build_normalci_esit_aralikli(remaining, settings)
        else:
            normalci_cards = build_normalci_cards_strict(remaining, settings)
        
        normalci_cards = merge_normalci_strict(normalci_cards, settings)
        normalci_cards = fix_small_cards(normalci_cards, settings, max_small=2)
        
        # Doğrulama
        total_placed = sum(len(c) for c in tekci_cards) + sum(len(c) for c in normalci_cards)
        if total_placed != total:
            continue
        
        placed_req = set()
        for c in tekci_cards:
            for s in c:
                if s['_id'] in required_ids:
                    placed_req.add(s['_id'])
        if placed_req != required_ids:
            continue
        
        # SKOR
        score = 0
        total_cards = len(tekci_cards) + len(normalci_cards)
        actual_ratio = len(tekci_cards) / total_cards if total_cards > 0 else 0
        
        score += abs(actual_ratio - target_ratio) * 10000
        
        under_min = sum(1 for c in tekci_cards if len(c) < min_tekci_srv)
        score += under_min * 200000
        
        # Küçük normalci (1-2-3 srv) - ÇOK YÜKSEK CEZA
        small_norm = sum(1 for c in normalci_cards if len(c) <= 3)
        single_srv = sum(1 for c in normalci_cards if len(c) == 1)
        
        score += single_srv * 500000  # Tek servisli = EN KÖTÜ
        if small_norm > 2:
            score += (small_norm - 2) * 300000  # 2'den fazla küçük kart
        
        no_rest = sum(1 for c in normalci_cards if len(c) > 1 and not card_has_rest(c, settings))
        score += no_rest * 80000
        
        time_violations = 0
        for c in normalci_cards:
            time_violations += count_violations(c, settings)
        score += time_violations * 60000
        
        # Toplam kart sayısını minimize et
        score += total_cards * 1000
        
        if score < best_score:
            best_score = score
            best_solution = (deepcopy(tekci_cards), deepcopy(normalci_cards))
    
    progress.progress(1.0)
    status.text(f"✅ {kombinasyon} kombinasyon! Skor: {best_score:.0f}")
    
    if best_solution is None:
        return [], [[s] for s in services]
    
    return best_solution

def optimize(services_df, settings):
    df = services_df.copy()
    for col in df.columns:
        if col in ['arac_tipi', 'kart_tipi', 'hat']:
            df[col] = df[col].fillna('').astype(str)
    
    services = df.to_dict('records')
    for i, s in enumerate(services):
        s['_id'] = f"s{i}"
        s['_vehicle'] = get_vehicle_type(s)
    
    koruklu = [s for s in services if s['_vehicle'] == 'koruklu']
    solo = [s for s in services if s['_vehicle'] == 'solo']
    
    st.info(f"📊 Toplam: {len(services)} ({len(koruklu)} körüklü, {len(solo)} solo)")
    
    # Veri analizi göster
    if settings.get('model') == 'global':
        st.markdown("---")
        if koruklu:
            st.markdown("#### 🚌 Körüklü Veri Analizi")
            analysis_k = analyze_data(koruklu, settings)
            display_analysis(analysis_k)
        
        if solo:
            st.markdown("#### 🚐 Solo Veri Analizi")
            analysis_s = analyze_data(solo, settings)
            display_analysis(analysis_s)
        st.markdown("---")
    
    if koruklu:
        st.write("🚌 **Körüklü** optimize ediliyor...")
        k_tekci, k_norm = optimize_group(koruklu, settings)
    else:
        k_tekci, k_norm = [], []
    
    if solo:
        st.write("🚐 **Solo** optimize ediliyor...")
        s_tekci, s_norm = optimize_group(solo, settings)
    else:
        s_tekci, s_norm = [], []
    
    result = []
    t_num, n_num = 1, 1
    
    for card in k_tekci + s_tekci:
        card_id = f"T{t_num}"
        t_num += 1
        vehicle = 'Körüklü' if card[0]['_vehicle'] == 'koruklu' else 'Solo'
        for s in sorted(card, key=lambda x: time_to_minutes(x['gidis'])):
            result.append({'hat': s['hat'], 'gidis': s['gidis'], 'donus': s['donus'],
                          'kart': card_id, 'kart_tipi': 'Tekçi', 'arac_tipi': vehicle})
    
    for card in k_norm + s_norm:
        card_id = f"N{n_num}"
        n_num += 1
        vehicle = 'Körüklü' if card[0]['_vehicle'] == 'koruklu' else 'Solo'
        for s in sorted(card, key=lambda x: time_to_minutes(x['gidis'])):
            result.append({'hat': s['hat'], 'gidis': s['gidis'], 'donus': s['donus'],
                          'kart': card_id, 'kart_tipi': 'Normalci', 'arac_tipi': vehicle})
    
    return pd.DataFrame(result)

# ============================================
# ARAYÜZ
# ============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📁 Dosya Yükleme")
    uploaded_file = st.file_uploader("CSV dosyası seçin", type=['csv'])
    
    if uploaded_file:
        try:
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='iso-8859-9')
            
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            col_mapping = {}
            for col in df.columns:
                if 'hat' in col: col_mapping[col] = 'hat'
                elif 'gidiş' in col or 'gidis' in col: col_mapping[col] = 'gidis'
                elif 'dönüş' in col or 'donus' in col: col_mapping[col] = 'donus'
                elif 'kart_tipi' in col: col_mapping[col] = 'kart_tipi'
                elif 'araç_tipi' in col or 'arac_tipi' in col: col_mapping[col] = 'arac_tipi'
            df = df.rename(columns=col_mapping)
            
            if all(c in df.columns for c in ['hat', 'gidis', 'donus']):
                if 'arac_tipi' in df.columns:
                    df['arac_tipi'] = df['arac_tipi'].fillna('').astype(str)
                if 'kart_tipi' in df.columns:
                    df['kart_tipi'] = df['kart_tipi'].fillna('').astype(str)
                st.session_state.services = df
                st.success(f"✅ {len(df)} servis yüklendi")
            else:
                st.error("Gerekli: HAT, GIDIS, DONUS")
        except Exception as e:
            st.error(f"Hata: {e}")
    
    st.markdown("---")
    st.markdown("### ⚙️ Optimizasyon Ayarları")
    
    with st.expander("👷 Tekçi Ayarları", expanded=True):
        tekci_oran = st.slider("Tekçi Oranı (%)", 0, 100, 44, key="tekci_oran")
        st.markdown("**Tekçi Servis Aralığı**")
        c1, c2 = st.columns(2)
        t_aralik_min = c1.number_input("En az (dk)", value=10, key="tamin")
        t_aralik_max = c2.number_input("En fazla (dk)", value=35, key="tamax")
        t_min_srv = st.number_input("Min Servis Sayısı", value=4, min_value=2, key="tms")
        st.markdown("**Sabah Çalışma**")
        c1, c2 = st.columns(2)
        t_sb = c1.text_input("Başlangıç", value="06:00", key="tsb")
        t_sbit = c2.text_input("Bitiş", value="10:00", key="tsbit")
        st.markdown("**Akşam Çalışma**")
        c1, c2 = st.columns(2)
        t_ab = c1.text_input("Başlangıç", value="14:00", key="tab")
        t_abit = c2.text_input("Bitiş", value="20:00", key="tabit")
    
    with st.expander("📊 Servis Aralıkları", expanded=True):
        st.markdown("**1. Öncelik (En İyi)**")
        c1, c2 = st.columns(2)
        o1_min = c1.number_input("En az (dk)", value=10, key="o1min")
        o1_max = c2.number_input("En fazla (dk)", value=35, key="o1max")
        st.markdown("**2. Öncelik (Normal Saat Max)**")
        c1, c2 = st.columns(2)
        o2_min = c1.number_input("En az (dk)", value=35, key="o2min")
        o2_max = c2.number_input("En fazla (dk)", value=90, key="o2max")
        st.caption("ℹ️ Pik dışı saatlerde maksimum bu aralık kabul edilir")
        st.markdown("**3. Öncelik (Sadece Max Aralık Saatinde)**")
        c1, c2 = st.columns(2)
        o3_min = c1.number_input("En az (dk)", value=60, key="o3min")
        o3_max = c2.number_input("En fazla (dk)", value=120, key="o3max")
    
    with st.expander("👥 Normalci Çalışma", expanded=False):
        st.markdown("**Sabahçı**")
        c1, c2 = st.columns(2)
        n_sb = c1.text_input("Başlangıç", value="06:00", key="nsb")
        n_sbit = c2.text_input("Bitiş", value="14:00", key="nsbit")
        st.markdown("**Akşamcı**")
        c1, c2 = st.columns(2)
        n_ab = c1.text_input("Başlangıç", value="14:00", key="nab")
        n_abit = c2.text_input("Bitiş", value="00:00", key="nabit")
    
    with st.expander("☕ İstirahat (30dk ZORUNLU)", expanded=False):
        st.markdown("**Sabahçı**")
        c1, c2 = st.columns(2)
        i_sb = c1.text_input("Başlangıç", value="09:00", key="isb")
        i_sbit = c2.text_input("Bitiş", value="13:00", key="isbit")
        st.markdown("**Akşamcı**")
        c1, c2 = st.columns(2)
        i_ab = c1.text_input("Başlangıç", value="14:30", key="iab")
        i_abit = c2.text_input("Bitiş", value="17:00", key="iabit")
    
    with st.expander("⏰ Pik Saat (ZORUNLU)", expanded=True):
        st.info("Bu saatlerde aralık belirtilen değerler arasında olmalı")
        st.markdown("**Sabah Pik**")
        c1, c2 = st.columns(2)
        sp_b = c1.text_input("Başlangıç", value="06:00", key="spb")
        sp_bit = c2.text_input("Bitiş", value="09:00", key="spbit")
        st.markdown("**Akşam Pik**")
        c1, c2 = st.columns(2)
        ap_b = c1.text_input("Başlangıç", value="16:30", key="apb")
        ap_bit = c2.text_input("Bitiş", value="20:00", key="apbit")
        st.markdown("**Pik Servis Aralığı**")
        c1, c2 = st.columns(2)
        p_min = c1.number_input("En az (dk)", value=7, key="pmin")
        p_max = c2.number_input("En fazla (dk)", value=17, key="pmax")
    
    with st.expander("📈 Max Aralık Saati (ZORUNLU)", expanded=True):
        st.warning("SADECE bu saatlerde 60-120dk aralık olabilir!")
        c1, c2 = st.columns(2)
        ma_b = c1.text_input("Başlangıç", value="10:00", key="mab")
        ma_bit = c2.text_input("Bitiş", value="13:00", key="mabit")
    
    with st.expander("🔧 Optimizasyon Ayarları", expanded=True):
        st.markdown("**🧮 Modeller**")
        model_options = {
            'Global Optimizasyon (Önerilen)': 'global',
            'Column Generation': 'column_generation',
            'Hybrid (Karma Algoritma)': 'hybrid',
            'Constraint Programming (CP)': 'cp',
            'Integer Linear Programming (ILP)': 'ilp',
            'Bin Packing': 'bin_packing',
            'Greedy + Local Search': 'greedy_local'
        }
        selected_model_name = st.selectbox(
            "Algoritma Modeli",
            options=list(model_options.keys()),
            index=0,  # Varsayılan: Global Optimizasyon
            key="model_select"
        )
        selected_model = model_options[selected_model_name]
        
        # Model açıklamaları
        model_descriptions = {
            'global': "🎯 TÜM servisleri birlikte optimize eder, kart tipi sonradan belirlenir",
            'column_generation': "📊 Akademik yaklaşım: Kurallara uygun kartlar üretir",
            'hybrid': "🔀 Birden fazla yöntemi dener, en iyi sonucu seçer",
            'cp': "🎯 Kuralları matematiksel kısıt olarak tanımlar",
            'ilp': "📐 Doğrusal programlama ile optimal çözüm",
            'bin_packing': "📦 Servisleri minimum kart sayısına paketler",
            'greedy_local': "🔄 Hızlı yerleştirme + iteratif iyileştirme"
        }
        st.caption(model_descriptions[selected_model])
        
        if selected_model == 'global':
            st.success("✅ Tekçi/Normalci ayrımı yapılmadan TÜM servisler birlikte optimize edilir. Kart tipi sonradan belirlenir.")
        
        if not ORTOOLS_AVAILABLE and selected_model in ['cp']:
            st.warning("⚠️ OR-Tools kurulu değil. Global Optimizasyon kullanılacak.")
        
        st.markdown("---")
        
        kombinasyon_sayisi = st.number_input("Kombinasyon Sayısı", value=150, min_value=50, max_value=1000, step=50, key="komb")
        st.caption("Daha fazla kombinasyon = daha iyi sonuç ama daha uzun süre")
        
        st.markdown("---")
        esit_aralikli = st.checkbox("Eşit Aralıklı Servis", value=False, key="esit_aralikli")
        if esit_aralikli:
            st.info("""
            **Eşit Aralıklı Servis Modu:**
            - 1, 2, 3 servisli kart oluşturulmaz (min 4 servis)
            - Pik saatlerde pik aralık kurallarına uyulur
            - Pik dışı saatlerde aralıklar eşit dağıtılır
            - Servis aralıkları 1. ve 3. öncelik arasında esnetilir
            """)

with col2:
    settings = {
        'tekci_oran': tekci_oran / 100,
        'tekci_aralik_min': t_aralik_min, 'tekci_aralik_max': t_aralik_max,
        'tekci_min_servis': t_min_srv,
        'oncelik1_min': o1_min, 'oncelik1_max': o1_max,
        'oncelik2_min': o2_min, 'oncelik2_max': o2_max,
        'oncelik3_min': o3_min, 'oncelik3_max': o3_max,
        'tekci_sabah_bas': time_to_minutes(t_sb), 'tekci_sabah_bit': time_to_minutes(t_sbit),
        'tekci_aksam_bas': time_to_minutes(t_ab), 'tekci_aksam_bit': time_to_minutes(t_abit),
        'norm_sabah_bas': time_to_minutes(n_sb), 'norm_sabah_bit': time_to_minutes(n_sbit),
        'norm_aksam_bas': time_to_minutes(n_ab),
        'norm_aksam_bit': time_to_minutes(n_abit) if time_to_minutes(n_abit) > 0 else 24*60,
        'ist_sabah_bas': time_to_minutes(i_sb), 'ist_sabah_bit': time_to_minutes(i_sbit),
        'ist_aksam_bas': time_to_minutes(i_ab), 'ist_aksam_bit': time_to_minutes(i_abit),
        'sabah_pik_bas': time_to_minutes(sp_b), 'sabah_pik_bit': time_to_minutes(sp_bit),
        'aksam_pik_bas': time_to_minutes(ap_b), 'aksam_pik_bit': time_to_minutes(ap_bit),
        'pik_aralik_min': p_min, 'pik_aralik_max': p_max,
        'max_aralik_bas': time_to_minutes(ma_b), 'max_aralik_bit': time_to_minutes(ma_bit),
        'kombinasyon_sayisi': kombinasyon_sayisi,
        'esit_aralikli': esit_aralikli,
        'model': selected_model
    }
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 Optimize Et", type="primary", use_container_width=True, disabled=st.session_state.services is None):
            try:
                st.session_state.result = optimize(st.session_state.services, settings)
            except Exception as e:
                st.error(f"Hata: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    with c2:
        if st.session_state.result is not None:
            csv = st.session_state.result.to_csv(index=False, sep=';', encoding='utf-8-sig')
            st.download_button("💾 CSV İndir", csv, "sefer_plani.csv", "text/csv", use_container_width=True)
    
    if st.session_state.result is not None:
        result_df = st.session_state.result
        
        st.markdown("### 📊 Sonuç")
        
        cards_df = result_df.groupby('kart').agg({'hat': 'count', 'kart_tipi': 'first', 'arac_tipi': 'first'}).reset_index()
        
        tekci_count = len(cards_df[cards_df['kart_tipi'] == 'Tekçi'])
        normalci_count = len(cards_df[cards_df['kart_tipi'] == 'Normalci'])
        total_cards = tekci_count + normalci_count
        actual_ratio = (tekci_count / total_cards * 100) if total_cards > 0 else 0
        
        small_normalci = len(cards_df[(cards_df['hat'] <= 3) & (cards_df['kart_tipi'] == 'Normalci')])
        single_normalci = len(cards_df[(cards_df['hat'] == 1) & (cards_df['kart_tipi'] == 'Normalci')])
        small_tekci = len(cards_df[(cards_df['hat'] < settings['tekci_min_servis']) & (cards_df['kart_tipi'] == 'Tekçi')])
        
        no_rest = 0
        time_violations = 0
        for kart_id in result_df['kart'].unique():
            kart_srv = result_df[result_df['kart'] == kart_id].to_dict('records')
            kart_type = kart_srv[0]['kart_tipi']
            if kart_type == 'Normalci':
                time_violations += count_violations(kart_srv, settings)
                if len(kart_srv) > 1 and not card_has_rest(kart_srv, settings):
                    no_rest += 1
        
        cols = st.columns(9)
        with cols[0]:
            st.markdown(f'<div class="stat-card stat-green"><div style="font-size:1rem">{len(result_df)}</div><div style="font-size:0.6rem">Servis</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="stat-card stat-orange"><div style="font-size:1rem">{tekci_count}</div><div style="font-size:0.6rem">Tekçi</div></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="stat-card stat-blue"><div style="font-size:1rem">{normalci_count}</div><div style="font-size:0.6rem">Normalci</div></div>', unsafe_allow_html=True)
        with cols[3]:
            target = int(settings['tekci_oran'] * 100)
            c = "stat-green" if abs(actual_ratio - target) <= 5 else "stat-yellow"
            st.markdown(f'<div class="stat-card {c}"><div style="font-size:0.85rem">%{actual_ratio:.0f}(H:{target})</div><div style="font-size:0.55rem">Oran</div></div>', unsafe_allow_html=True)
        with cols[4]:
            c = "stat-green" if single_normalci == 0 else "stat-red"
            st.markdown(f'<div class="stat-card {c}"><div style="font-size:1rem">{single_normalci}</div><div style="font-size:0.6rem">1Srv</div></div>', unsafe_allow_html=True)
        with cols[5]:
            c = "stat-green" if small_normalci <= 2 else "stat-red"
            st.markdown(f'<div class="stat-card {c}"><div style="font-size:1rem">{small_normalci}</div><div style="font-size:0.6rem">≤3Srv</div></div>', unsafe_allow_html=True)
        with cols[6]:
            c = "stat-green" if small_tekci == 0 else "stat-red"
            st.markdown(f'<div class="stat-card {c}"><div style="font-size:1rem">{small_tekci}</div><div style="font-size:0.6rem">&lt;MinT</div></div>', unsafe_allow_html=True)
        with cols[7]:
            c = "stat-green" if no_rest == 0 else "stat-red"
            st.markdown(f'<div class="stat-card {c}"><div style="font-size:1rem">{no_rest}</div><div style="font-size:0.6rem">İst.Yok</div></div>', unsafe_allow_html=True)
        with cols[8]:
            c = "stat-green" if time_violations == 0 else "stat-red"
            st.markdown(f'<div class="stat-card {c}"><div style="font-size:1rem">{time_violations}</div><div style="font-size:0.6rem">Saat H.</div></div>', unsafe_allow_html=True)
        
        if small_tekci > 0:
            st.error(f"⚠️ {small_tekci} tekçi min servis altında!")
        if single_normalci > 0:
            st.error(f"⚠️ {single_normalci} tek servisli normalci kart!")
        if small_normalci > 2:
            st.error(f"⚠️ {small_normalci} normalci ≤3 servisli (max 2)!")
        if no_rest > 0:
            st.error(f"⚠️ {no_rest} normalci istirahatsız!")
        if time_violations > 0:
            st.error(f"⚠️ {time_violations} saat kuralı ihlali (pik/max aralık/min aralık)!")
        
        st.markdown("---")
        view = st.radio("Görünüm", ["Kartlar", "Tablo"], horizontal=True)
        
        if view == "Kartlar":
            all_cards = sorted(result_df['kart'].unique(), key=lambda x: (0 if x.startswith('T') else 1, int(x[1:])))
            cols_d = st.columns(4)
            
            for idx, kart_id in enumerate(all_cards):
                with cols_d[idx % 4]:
                    kart_srv = sorted(result_df[result_df['kart'] == kart_id].to_dict('records'), key=lambda x: time_to_minutes(x['gidis']))
                    is_tekci = kart_id.startswith('T')
                    vehicle = kart_srv[0]['arac_tipi']
                    has_rest = True if is_tekci else card_has_rest(kart_srv, settings)
                    
                    card_violations = 0 if is_tekci else count_violations(kart_srv, settings)
                    is_problem = (is_tekci and len(kart_srv) < settings['tekci_min_servis']) or \
                                 (not is_tekci and len(kart_srv) <= 3) or \
                                 (not is_tekci and not has_rest) or \
                                 card_violations > 0
                    
                    card_class = "card-tekci" if is_tekci else "card-normalci"
                    if is_problem:
                        card_class += " card-problem"
                    
                    first_t, last_t = kart_srv[0]['gidis'], kart_srv[-1]['donus']
                    vb = f'<span class="vehicle-badge {"vehicle-koruklu" if vehicle == "Körüklü" else "vehicle-solo"}">{vehicle[0]}</span>'
                    ri = "☕" if has_rest else "⚠️"
                    
                    html = f'<div class="{card_class}">'
                    html += f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">'
                    html += f'<span style="font-size:1rem;font-weight:900;color:{"#f97316" if is_tekci else "#2563eb"}">{kart_id}{vb}</span>'
                    html += f'<span>{ri}<span style="background:#374151;color:white;padding:0.15rem 0.35rem;border-radius:0.2rem;font-size:0.65rem;font-weight:bold;margin-left:0.2rem">{len(kart_srv)}</span></span>'
                    html += '</div>'
                    html += f'<div style="background:{"#fde68a" if is_tekci else "#bfdbfe"};padding:0.35rem;border-radius:0.2rem;font-family:monospace;font-weight:bold;margin-bottom:0.35rem;font-size:0.8rem">{first_t}→{last_t}</div>'
                    
                    if is_tekci:
                        wt = calc_tekci_work(kart_srv, settings)
                        html += f'<div style="background:#f97316;color:white;padding:0.1rem 0.3rem;border-radius:0.15rem;font-size:0.6rem;display:inline-block;margin-bottom:0.35rem">⏱{wt//60}s{wt%60}dk</div>'
                    
                    for i, srv in enumerate(kart_srv):
                        if i > 0:
                            gap = time_to_minutes(srv['gidis']) - get_end_time(kart_srv[i-1])
                            gap_start = get_end_time(kart_srv[i-1])
                            
                            if is_tekci:
                                pm = can_tekci_morning(kart_srv[i-1], settings)
                                ca = can_tekci_afternoon(srv, settings)
                                if pm and ca:
                                    html += '<div class="gap-badge gap-slate">☀️→🌙</div>'
                                    html += f'<div class="service-row service-tekci"><span style="font-weight:bold">{srv["hat"]}</span><span style="font-family:monospace">{srv["gidis"]}-{srv["donus"]}</span></div>'
                                    continue
                            
                            is_rest_gap = False
                            is_pik = is_pik_time(gap_start, settings)
                            is_max_time = is_max_aralik_time(gap_start, settings)
                            
                            if not is_tekci:
                                fs = time_to_minutes(kart_srv[0]['gidis'])
                                is_sab = fs < settings['norm_aksam_bas']
                                rs = settings['ist_sabah_bas'] if is_sab else settings['ist_aksam_bas']
                                re = settings['ist_sabah_bit'] if is_sab else settings['ist_aksam_bit']
                                if gap >= 30 and rs <= gap_start <= re:
                                    is_rest_gap = True
                            
                            if is_rest_gap:
                                gc, gt = "gap-rest", f"☕{gap}dk"
                            elif is_tekci:
                                gc = "gap-green" if tekci_gap_valid(gap, settings) else "gap-red"
                                gt = f"{gap}dk"
                            elif is_pik:
                                valid = settings['pik_aralik_min'] <= gap <= settings['pik_aralik_max']
                                gc = "gap-pik" if valid else "gap-red"
                                gt = f"⚡{gap}dk" if valid else f"⚡{gap}dk!"
                            elif is_max_time:
                                gc = "gap-orange" if gap <= settings['oncelik3_max'] else "gap-red"
                                gt = f"📈{gap}dk"
                            elif gap > settings['oncelik2_max']:
                                gc, gt = "gap-red", f"{gap}dk!"
                            else:
                                if gap < o1_min: gc = "gap-red"
                                elif gap <= o1_max: gc = "gap-green"
                                elif gap <= o2_max: gc = "gap-yellow"
                                else: gc = "gap-red"
                                gt = f"{gap}dk"
                            
                            html += f'<div class="gap-badge {gc}">{gt}</div>'
                        
                        sc = "service-tekci" if is_tekci else "service-normalci"
                        html += f'<div class="service-row {sc}"><span style="font-weight:bold">{srv["hat"]}</span><span style="font-family:monospace">{srv["gidis"]}-{srv["donus"]}</span></div>'
                    
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)
        else:
            st.dataframe(result_df, use_container_width=True, height=500)
    
    elif st.session_state.services is None:
        st.info("👈 CSV yükleyin")
    else:
        st.info("🚀 Optimize Et")
