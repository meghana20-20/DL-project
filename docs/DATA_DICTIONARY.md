# Unified Marine & Oceanographic Data Platform: Data Dictionary

This document defines the schema, units, statistical bounds, physical validation rules, and imputation strategies for all variables within the unified platform.

---

## 1. Physical Oceanographic Sensor Measurements (CalCOFI Modality)

| Variable Identifier | Display Name | Unit | Type | Physical Range | Domain Validation Rule & Imputation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `station_id` | Hydrographic Station ID | String | String | Nominal | Unique identifier for the geographical ocean cast station. |
| `latitude` | Station Latitude | Decimal Degrees | Float64 | $[-90.0, +90.0]$ | Northern/Southern coordinate. Must be physically valid geographic latitude. |
| `longitude` | Station Longitude | Decimal Degrees | Float64 | $[-180.0, +180.0]$ | Eastern/Western coordinate. Validated within marine geographic boundaries. |
| `depth_m` | Sampling Depth | Meters ($m$) | Float64 | $[0.0, 5000.0]$ | Hydrostatic depth. Values $< 0$ flagged as sensor anomalies and dropped. |
| `water_temp_c` | Seawater Temperature | Celsius ($^\circ\text{C}$) | Float64 | $[-2.5, 36.0]$ | Thermodynamic limits of ocean water. Standard oceanographic sensor measurements. |
| `salinity_psu` | Practical Salinity | Practical Salinity Units (PSU) | Float64 | $[24.0, 42.0]$ | Haline range for coastal/open-ocean seawater. Values outside interval quarantined. |
| `dissolved_oxygen_ml_l` | Dissolved Oxygen Concentration | Milliliters per Liter ($ml/L$) | Float64 | $[0.0, 12.0]$ | Critical for hypoxia analysis. Levels $< 1.4\text{ ml/L}$ designate severe oxygen minimum zones. |
| `phosphate_umol_l` | Orthophosphate Concentration ($PO_4$) | $\mu\text{mol}/L$ | Float64 | $[0.0, 5.0]$ | Geochemical macro-nutrient. Depth-stratified median imputation with indicator flag. |
| `nitrate_umol_l` | Nitrate Concentration ($NO_3$) | $\mu\text{mol}/L$ | Float64 | $[0.0, 60.0]$ | Essential nutrient for phytoplankton. Depth-stratified median imputation. |
| `chlorophyll_a_ug_l` | Chlorophyll-a Biomass Proxy | $\mu g/L$ | Float64 | $[0.0, 25.0]$ | Primary productivity index. Euphotic zone concentrated; decays rapidly below 150m. |

---

## 2. Commercial Fisheries & Catch Activity (FAO Modality)

| Variable Identifier | Display Name | Unit | Type | Category / Range | Description & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `fao_zone_code` | FAO Major Fishing Area | Discrete Code | Integer / Category | $[10, 88]$ | Standardized Food and Agriculture Organization marine statistical area identifier. |
| `gear_type` | Commercial Fishing Gear | String / Token | String | Purse Seine, Trawl, Longline, Gillnet, Artisanal | Commercial extraction technology employed in the surveyed marine sector. |
| `fishing_effort_hours` | Fishing Effort Duration | Hours | Float64 | $[1.0, 1000.0]$ | Standardized operational fishing effort metric per quadrant. |
| `catch_tonnage_mt` | Annual Marine Catch Landing | Metric Tons ($MT$) | Float64 | $[0.0, 15000.0]$ | Reported commercial biomass harvest for the corresponding spatial quadrant. |

---

## 3. Marine Biodiversity & Taxonomic Modality (OBIS / OceanBench Modality)

| Variable Identifier | Display Name | Unit | Type | Category / Range | Description & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `species_taxon_id` | Taxonomic Species Key | String / Token | String | Categorical | Unique scientific binomen (e.g., *Engraulis mordax*, *Sardinops sagax*, *Thunnus alalunga*). |
| `taxonomic_family` | Marine Biological Family | String / Token | String | Clupeidae, Scombridae, Gadidae, etc. | Higher phylogenetic taxonomic grouping used for categorical embeddings. |
| `shannon_diversity_index` | Shannon-Wiener Diversity Metric | Index ($H'$) | Float64 | $[0.5, 4.5]$ | Alpha biodiversity index quantifying community evenness and taxonomic richness. |

---

## 4. Modeling Targets (Handover to Modeling Lead)

| Target Identifier | Type | Objective | Value Range / Classes | Modeling Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `target_abundance_density` | Continuous | Regression | $[0.0, 1200.0]$ (individuals / $1000m^3$) | Supervised regression baseline (Random Forest / Ridge) and 1D CNN / MLP deep learning target. |
| `target_vulnerability_tier` | Discrete | Multi-Class Classification | 0: Low, 1: Moderate, 2: High, 3: Critical | Supervised classification target assessing ecological conservation urgency under climate forcing. |

---

## 5. Derived Engineered Features

* `thermocline_gradient`: Rate of temperature decline with depth ($\Delta T / \Delta \text{Depth}$).
* `np_ratio`: Nitrate-to-Phosphate ratio ($NO_3 / PO_4$) measuring biological Redfield stoichiometry (limiting nutrient proxy).
* `hypoxia_flag`: Binary indicator (1 if `dissolved_oxygen_ml_l` $< 1.4\text{ ml/L}$, else 0).
* `temp_salinity_density_proxy`: Nonlinear seawater sigma-t density anomaly proxy computed from salinity and temperature.
