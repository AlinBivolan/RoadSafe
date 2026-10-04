RoadSafe — analiză, prevenție și risc rutier
RoadSafe este o aplicație dezvoltată în Streamlit pentru analiza accidentelor rutiere și identificarea zonelor cu risc ridicat. Proiectul pornește de la date istorice despre accidente și combină analiza statistică cu informații geografice pentru a oferi o imagine cât mai clară asupra riscului rutier.

Aplicația principală, app.py, se ocupă de încărcarea și curățarea datelor, analiza accidentelor și reprezentarea rezultatelor pe hărți interactive. Sunt analizate aspecte precum evoluția accidentelor în timp, orele și zilele în care apar cel mai frecvent, severitatea accidentelor, condițiile meteo, lumina și diferențele dintre zonele urbane și rurale.

Pentru identificarea zonelor cu risc este folosit un scor compozit care ține cont atât de numărul accidentelor, cât și de severitatea acestora și de proporția accidentelor grave. În cazul traseelor, scorul este ajustat pentru a pune mai mult accent pe severitate, astfel încât o zonă foarte aglomerată să nu fie considerată automat mai periculoasă doar pentru că înregistrează mai multe accidente.

O parte importantă a proiectului este componenta de prevenție. Rețeaua rutieră este obținută din OpenStreetMap, iar intersecțiile sunt identificate și asociate cu accidentele din dataset. Pentru căutarea rapidă a celei mai apropiate intersecții este utilizat un KDTree. Accidentele care nu pot fi asociate unei intersecții sunt analizate separat, în special în zonele rurale, unde riscul poate apărea pe segmente de drum, curbe sau porțiuni izolate.

Aplicația permite și evaluarea riscului pe trasee. Utilizatorul poate introduce punctul de plecare și destinația, iar aplicația obține ruta prin OSRM și calculează riscul pentru diferite segmente ale acesteia. Rezultatul este reprezentat direct pe hartă, astfel încât zonele mai riscante să poată fi observate ușor.

Pentru optimizarea calculelor am folosit în principal Joblib. Este potrivit pentru acest proiect deoarece permite paralelizarea unor calcule independente și se integrează bine cu biblioteci precum NumPy, GeoPandas, Shapely și SciPy. Am folosit și Dask, însă în proiect acesta are în principal rolul de a permite comparația dintre diferite metode de paralelizare.

Benchmark-urile au fost separate de aplicația principală în două aplicații independente. app_benchmark_section5.py testează partea de prevenție, iar app_benchmark_section6.py testează calcularea riscului pe trasee. Sunt comparate Joblib și Dask folosind diferite configurații de workeri, iar rezultatele sunt analizate prin timpul de execuție, speedup și eficiență.

Un aspect important este alegerea automată a numărului de workeri. Aplicația ține cont de numărul de procesoare, memoria disponibilă și dimensiunea datasetului, încercând să găsească un echilibru între performanță și stabilitatea aplicației. Mai mulți workeri nu înseamnă întotdeauna un timp mai mic de execuție, deoarece paralelizarea introduce și un anumit overhead.

Pentru a evita recalcularea unor operații costisitoare, proiectul folosește și mecanisme de cache. Acestea sunt utile în special pentru încărcarea datelor și pentru rețelele rutiere descărcate din OpenStreetMap.

Proiectul este organizat astfel încât aplicația principală, benchmark-urile și componentele de procesare să fie separate. Datele sunt păstrate în folderul data, componentele aplicației în src, iar rețelele rutiere cache-uite în cache. Pentru rulare este suficientă instalarea dependențelor din requirements.txt și pornirea aplicației cu streamlit run app.py.

RoadSafe are și câteva limitări. Serviciile precum OpenStreetMap, Nominatim și OSRM depind de conexiunea la internet, iar rezultatele sunt influențate de calitatea datelor despre accidente. Scorul de risc este un indicator construit pentru analiză și interpretare, nu un model predictiv antrenat pe date.

Pe viitor, proiectul poate fi extins prin adăugarea unui sistem persistent pentru cache-ul rutelor și geocodărilor, exportul rezultatelor, teste automate și configurarea mai flexibilă a ponderilor. O direcție interesantă ar fi și introducerea unui model de machine learning pentru predicția severității sau a riscului de accident.
