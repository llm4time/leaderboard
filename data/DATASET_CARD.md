## Datasets Description

We conducted experiments to evaluate the benchmark across seven real-world datasets spanning diverse applications, characteristics, and lengths. Although most datasets are multivariate (the Carbon Monitor dataset is an exception), we selected a single representative target variable from each dataset to frame the problem as a univariate time series forecasting task.

1. **ETTh2** — hourly time series from the energy domain with 17,420 observations representing electricity transformer operations.
2. **Electricity** — electricity consumption from 321 customers, covering the period from 2016 to 2019.
3. **Traffic** — urban mobility dataset comprising 5,088 records of bus line occupancy.
4. **COVID-19** — daily patient/case counts collected from COVID-19 centers across regions.
5. **Wike2000** — daily Wikipedia page views capturing user access patterns across multiple pages.
6. **Retail** — retail-sector dataset with 849 observations, representing commercial demand dynamics.
7. **Carbon Monitor** — 791 observations of daily CO₂ emissions. We derive time series by aggregating emissions across sectors for each country, and also across all countries and sectors. Original data cover nine regions (China, USA, India, EU, Russia, Japan, Brazil, UK, Rest of World) and six sectors (domestic aviation, ground transport, industry, international aviation, power, residential).

Perturbations and augmentation strategies used to mitigate contamination risks and reduce memorization:

- **Scaling:** multiply all values by a constant factor (scale = 1.10). Applied to Wike2000 and Retail to shift magnitude while preserving relative patterns.
- **Temporal shifting:** add one day to all timestamps to prevent models from relying on date-specific patterns. Applied to COVID-19.
- **Noise injection:** add white noise sampled from N(0, σ²), where σ = 0.03 · std(X) for each training or testing window. Noise is applied independently to training and testing splits for ETTh2 and Electricity to avoid leakage.
- **Magnitude warping:** apply smooth random variations to series magnitude via cubic-spline interpolation. Applied to Traffic with knot = 4 and σ = 0.2.

The Carbon Monitor dataset did not receive perturbations because it consists of derived aggregations not present in the original raw files. Values in the Traffic and Wike2000 datasets are truncated to preserve integer counts.

Notes:

- Citations and figure references were removed from this card; add references where appropriate.
- If you prefer a different markdown layout (tables, per-dataset metadata blocks, sample preview images), eu posso padronizar conforme o estilo desejado.
