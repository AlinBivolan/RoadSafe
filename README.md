# RoadSafe — analiză, prevenție și risc rutier

RoadSafe este o aplicație Streamlit pentru analiza accidentelor rutiere, identificarea zonelor cu risc și evaluarea riscului pe trasee. Proiectul combină date istorice despre accidente, hărți interactive, intersecții OpenStreetMap, geocodare, rute OSRM și aplicații separate de benchmark.

## Ce face aplicația principală

Aplicația principală este `app.py`. Ea este gândită pentru utilizare normală și prezentare, fără benchmark la final.

1. **Încarcă și curăță datasetul de accidente**
   - validează coloanele obligatorii;
   - convertește coordonate, severitate, victime, viteză, dată și oră;
   - creează variabile utile: an, lună, oră, zi, tip zonă, severitate, accident grav.

2. **Analiză exploratorie**
   - accidente pe ani, ore și zile;
   - distribuție pe severitate, vreme, lumină și urban/rural;
   - hărți interactive cu puncte și grilă de risc.

3. **Scor compozit de risc**
   - pentru grid și prevenție: `35% frecvență + 40% severitate + 25% pondere accidente grave`;
   - pentru trasee: accent mai mare pe severitate, ca să nu fie supraevaluate zonele urbane doar pentru că au multe accidente.

4. **Prevenție**
   - descarcă rețeaua rutieră din OpenStreetMap;
   - extrage intersecții reale;
   - asociază accidentele la intersecții prin KDTree și raze de asociere;
   - detectează hotspot-uri rurale pentru accidente neasociate intersecțiilor.

5. **Risc pe traseu**
   - geocodează start/destinație;
   - obține rute OSRM;
   - calculează risc global și risc pe segmente;
   - colorează traseul pe hartă în funcție de risc.

## Aplicații separate de benchmark

Benchmark-ul a fost scos din aplicația principală și mutat în aplicații independente:

```bash
streamlit run app_benchmark_section5.py
streamlit run app_benchmark_section6.py
```

- `app_benchmark_section5.py` testează partea de prevenție: asociere accidente la intersecții și hotspot-uri rurale.
- `app_benchmark_section6.py` testează partea de traseu: risc pe segmente și puncte critice.

Aceste aplicații compară:

- Joblib;
- Dask;
- număr diferit de workeri;
- speedup față de baseline;
- eficiență procentuală.

Pentru demonstrații rapide, benchmark-urile pot rula în mod sintetic offline, fără OSM/OSRM. Pentru teste mai realiste, se poate folosi OpenStreetMap sau OSRM, dar acestea depind de internet.

## De ce aplicația principală folosește Joblib

Pentru calculele geospațiale din proiect, Joblib este alegerea implicită, deoarece:

- este simplu și stabil în Streamlit;
- merge bine cu `GeoPandas`, `Shapely`, `SciPy cKDTree` și `NumPy`;
- are overhead mai mic decât Dask pentru multe calcule locale;
- este potrivit pentru paralelizare pe segmente sau pe operații independente.

Dask rămâne util pentru studiu, demonstrație și dataseturi foarte mari, dar nu este obligatoriu pentru rularea normală a aplicației.

## Explicația modurilor de optimizare

### Joblib

Joblib permite rularea unei funcții de mai multe ori în paralel, pe mai mulți workeri. În proiect este folosit pentru:

- împărțirea calculelor pe segmente de traseu;
- rularea unor calcule independente în paralel;
- folosirea mai multor thread-uri în funcțiile unde datele sunt deja în memorie.

Este potrivit când ai calcule locale și nu vrei să gestionezi un cluster sau un scheduler complex.

### Dask

Dask este un framework pentru calcule pe task-uri și bucăți de date. Poate lucra local sau distribuit pe mai multe mașini. În benchmark-urile proiectului este folosit pentru a demonstra:

- împărțirea unui dataframe în bucăți;
- programarea calculelor ca task-uri;
- comparația între scheduler-ul Dask și Joblib.

Pentru dataseturi mici sau calcule geospațiale cu multe obiecte geometrice, Dask poate fi mai lent din cauza overhead-ului.

### Workeri

Un worker este o unitate de execuție care poate lucra în paralel cu altele. Mai mulți workeri pot reduce timpul, dar nu mereu. Dacă datasetul este mic, dacă RAM-ul este limitat sau dacă task-urile copiază multă geometrie, prea mulți workeri pot încetini aplicația.

### Paralelism pe segmente

În secțiunea 6, ruta este împărțită în segmente. Fiecare segment poate fi analizat independent, deci este o problemă potrivită pentru paralelizare.

### KDTree

KDTree accelerează căutarea celui mai apropiat punct. În secțiunea 5, în loc să comparăm fiecare accident cu fiecare intersecție, construim un arbore spațial și căutăm rapid cea mai apropiată intersecție.

### Vectorizare

Vectorizarea înseamnă să evităm buclele Python rând-cu-rând și să lucrăm cu operații NumPy/Pandas pe coloane întregi. Este una dintre cele mai importante optimizări pentru secțiunea 5.

### Cache

Streamlit cache-uiește datele și operațiile scumpe. De exemplu, încărcarea datasetului și descărcarea intersecțiilor OSM nu trebuie refăcute la fiecare interacțiune.

## Auto workers: cum funcționează

Aplicația estimează automat configurarea potrivită pentru calculatorul pe care rulează:

- verifică numărul de CPU-uri logice;
- încearcă să detecteze CPU-urile fizice cu `psutil`;
- verifică RAM-ul disponibil;
- ține cont de numărul de rânduri din datasetul filtrat;
- limitează conservator numărul de workeri ca să nu blocheze interfața Streamlit.

Regula importantă este că **mai mulți workeri nu înseamnă automat mai rapid**. Pentru dataseturi mici, overhead-ul de paralelizare poate face rularea mai lentă. De aceea modul automat preferă valori moderate, iar benchmark-urile separate pot confirma alegerea.

## Structură proiect

```text
roadsafe_improved/
  app.py
  app_benchmark_section5.py
  app_benchmark_section6.py
  requirements.txt
  README.md
  src/
    __init__.py
    analytics.py
    benchmark.py
    data_loader.py
    maps.py
    parallel_prevention.py
    parallel_risk.py
    performance.py
    prevention.py
    risk.py
    risk_scoring.py
    road_network.py
    routing.py
    theme.py
  data/
    accidents.csv
    uk_places_official.csv
  assets/
    styles.css
  cache/
    road_networks/
  .streamlit/
    config.toml
```

## Instalare

Creează un mediu virtual:

```bash
python -m venv .venv
```

Activează mediul:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Instalează dependențele:

```bash
pip install -r requirements.txt
```

## Date necesare

Pune datasetul principal aici:

```text
data/accidents.csv
```

Opțional, pentru catalogul de zone folosit la prevenție:

```text
data/uk_places_official.csv
```

Aplicația se așteaptă ca `accidents.csv` să conțină cel puțin coloanele:

```text
latitude
longitude
collision_severity
number_of_casualties
speed_limit
date
time
```

Coloane opționale, dar utile:

```text
urban_or_rural_area
weather_conditions
light_conditions
```

## Rulare

Aplicația principală:

```bash
streamlit run app.py
```

Benchmark secțiunea 5:

```bash
streamlit run app_benchmark_section5.py
```

Benchmark secțiunea 6:

```bash
streamlit run app_benchmark_section6.py
```

## Recomandare pentru prezentare

Pentru demonstrație, pregătește din timp:

1. datasetul în `data/accidents.csv`;
2. o zonă OSM deja cache-uită în `cache/road_networks/`;
3. unul sau două trasee testate;
4. rezultate benchmark salvate pentru 1, 2, 4 și 8 workeri.

Astfel aplicația va rula mai stabil chiar dacă internetul este lent.

## Limitări

- OSMnx, Nominatim și OSRM depind de servicii externe și de conexiunea la internet;
- scorul de risc este interpretabil, nu un model predictiv antrenat;
- precizia depinde de calitatea coordonatelor din dataset;
- proiecția metrică EPSG:3857 este practică pentru calcule locale, dar nu este perfectă pentru distanțe foarte mari;
- Dask poate fi mai lent decât Joblib pe dataseturi mici sau task-uri geospațiale cu overhead mare.

## Direcții viitoare

- cache persistent pentru geocodări și rute;
- export CSV/HTML pentru hărți și tabele;
- teste unitare pentru funcțiile de scor;
- configurare YAML pentru ponderi și praguri;
- model ML pentru predicție de severitate sau risc.

## Clarificare: secțiunea 5 - intersecții și hotspot-uri rurale

În secțiunea 5, aplicația tratează separat două tipuri de risc rutier:

1. **Accidente asociate intersecțiilor** - pentru fiecare accident se caută cea mai apropiată intersecție reală extrasă din OpenStreetMap. Accidentul este atribuit intersecției doar dacă se află în raza de asociere a acelei intersecții.
2. **Hotspot-uri rurale** - accidentele care nu au fost asociate unei intersecții sunt analizate separat, mai ales în zone rurale, unde riscul apare des pe segmente de drum, curbe sau porțiuni izolate, nu neapărat în intersecții.

Când utilizatorul introduce un nume de loc, de exemplu `London, England`, aplicația nu caută pe o rază fixă. OSMnx identifică zona administrativă disponibilă în OpenStreetMap și descarcă rețeaua rutieră de tip `drive` din acea zonă. După aceea, fiecare intersecție primește o rază de asociere: standard 50 m, dar redusă la jumătate din distanța față de cea mai apropiată intersecție atunci când intersecțiile sunt mai apropiate de 100 m.

```text
raza = min(50 m, jumătate din distanța până la cea mai apropiată intersecție)
```

Harta de prevenție se centrează automat pe zona analizată. În tabelul cu top intersecții, utilizatorul poate selecta o intersecție, iar harta se mută pe punctul respectiv.
