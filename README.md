# Global Superstore — Revival Strategy Dashboard

## 1. Install
```bash
pip install -r requirements.txt
```

## 2. Run
Pastikan `Global_Superstore2.csv` berada satu folder dengan `app.py`.

```bash
streamlit run app.py
```

## 3. Gemini API Key
Pilihan paling aman untuk lokal:

Windows PowerShell:
```powershell
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
streamlit run app.py
```

Atau masukkan key langsung pada halaman **AI Recommendation**.

Untuk Streamlit Cloud, gunakan **Settings → Secrets**:
```toml
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

## 4. Struktur dashboard
- Performance: Sales, Profit, Margin, YoY, monthly trend
- Geography: Market → Region → Country
- Product: Category → Sub-Category → Product
- Root Cause: Discount, Shipping Cost, Ship Mode, loss concentration
- AI Recommendation: LangChain + Gemini menghasilkan executive message, findings, actions, owner, timeline, KPI, dan expected impact

## Catatan analisis
Dashboard membedakan pertumbuhan perusahaan dari kualitas profitabilitas. Hubungan discount dan profit adalah pola observasional, bukan bukti kausalitas; gunakan A/B test atau pilot pricing untuk memvalidasi kebijakan baru.
