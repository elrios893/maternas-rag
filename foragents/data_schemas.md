# Data Schemas — Maternas Datasets

Documento de referencia para planificación con Claude Opus.  
Contiene el esquema de cada dataset con 10 valores representativos de muestra.

---

## 1. Dataset: `data/`

**Descripción:** Preguntas de opción múltiple de exámenes médicos indios (estilo USMLE/PGI/AIIMS), con explicaciones detalladas de cada respuesta.

**Formato:** JSONL — un objeto JSON por línea  
**Archivos:** `data/train.json` · `data/dev.json` · `data/test.json`  
**Tamaño:** train ≈ 182 k registros · dev ≈ 4 183 · test ≈ 6 150  
**Idioma:** Inglés

### Esquema de campos

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | `string` (UUID) | Identificador único de la pregunta |
| `question` | `string` | Texto de la pregunta médica |
| `opa` | `string` | Opción A |
| `opb` | `string` | Opción B |
| `opc` | `string` | Opción C |
| `opd` | `string` | Opción D |
| `cop` | `integer` (1–4) | Índice de la opción correcta (1=opa, 2=opb, 3=opc, 4=opd) |
| `choice_type` | `string` | `"single"` o `"multi"` |
| `exp` | `string` | Explicación / justificación de la respuesta correcta |
| `subject_name` | `string` | Asignatura médica (Anatomy, Biochemistry, Surgery…) |
| `topic_name` | `string \| null` | Subtema dentro de la asignatura |

### 10 muestras representativas

| # | id (abrev.) | question (resumen) | cop | subject_name | topic_name | choice_type |
|---|-------------|-------------------|-----|-------------|------------|-------------|
| 1 | e9ad821a | Chronic urethral obstruction due to benign prismatic hyperplasia can lead to... | 3 | Anatomy | Urinary tract | single |
| 2 | e3d3c4e1 | Which vitamin is supplied from only animal source? | 3 | Biochemistry | Vitamins and Minerals | single |
| 3 | 5c38bea6 | All of the following are surgical options for morbid obesity except – | 4 | Surgery | Surgical Treatment Obesity | multi |
| 4 | cdeedb04 | Following endarterectomy on the right common carotid, a patient is found blind in the right eye... | 1 | Ophthalmology | *(null)* | multi |
| 5 | dc6794a3 | Growth hormone has its effect on growth through? | 2 | Physiology | *(null)* | single |
| 6 | 5ab84ea8 | Scrub typhus is transmitted by: September 2004 | 3 | Social & Preventive Medicine | *(null)* | single |
| 7 | f3bf8583 | Per rectum examination is not a useful test for diagnosis of | 3 | Surgery | Urology | single |
| 8 | 53f79833 | Hypomimia is? | 3 | Psychiatry | *(null)* | single |
| 9 | e529be7c | Which of the following statements are True/False about Hirsutism? (5 enunciados) | 3 | Medicine | *(null)* | multi |
| 10 | d64eabcf | True regarding lag phase is? | 1 | Microbiology | general microbiology | multi |

### Ejemplo completo (registro #1)
```json
{
  "id": "e9ad821a-c438-4965-9f77-760819dfa155",
  "question": "Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma",
  "opa": "Hyperplasia",
  "opb": "Hyperophy",
  "opc": "Atrophy",
  "opd": "Dyplasia",
  "cop": 3,
  "choice_type": "single",
  "exp": "Chronic urethral obstruction because of urinary calculi, prostatic hypertrophy... cause hydronephrosis... Refer Robbins 7th/9e p950",
  "subject_name": "Anatomy",
  "topic_name": "Urinary tract"
}
```

---

## 2. Dataset: `data_clean/`

**Descripción:** Versión limpia y multilingüe del dataset MedQA, con preguntas de exámenes médicos de tres regiones y textbooks médicos de referencia.

**Formato:** JSONL (preguntas) + texto plano (libros de texto)  
**Idiomas:** Inglés (US), Chino tradicional (Taiwan), Chino simplificado (Mainland)

### Estructura de directorios

```
data_clean/data_clean/
├── questions/
│   ├── Mainland/   → Examen nacional chino MCMLE (chino simplificado, 5 opciones A-E)
│   ├── Taiwan/     → Examen médico taiwanés (chino tradicional, 4 opciones A-D)
│   └── US/         → USMLE Steps 1, 2 & 3 (inglés, 5 opciones A-E)
│       ├── train.jsonl / dev.jsonl / test.jsonl
└── textbooks/
    ├── en/             → 18 libros en inglés (Gray's Anatomy, Harrison, Robbins…)
    ├── zh_paragraph/   → Libros chinos partidos por párrafo
    └── zh_sentence/    → Libros chinos partidos por oración
```

### Esquema de campos — Questions

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `question` | `string` | Texto de la pregunta |
| `answer` | `string` | Texto completo de la respuesta correcta |
| `options` | `object` | Mapa letra → texto (`{"A": "...", "B": "...", ...}`) |
| `meta_info` | `string` | Origen del examen (`step1`, `step2&3`, `taiwanese_test_Q`, nombre de sección china) |
| `answer_idx` | `string` | Letra de la respuesta correcta (`"A"`–`"E"`) |

### Esquema de campos — Textbooks (`en/`)

| Campo | Descripción |
|-------|-------------|
| Archivo `.txt` | Texto completo del libro, sin estructura JSON. Un archivo = un libro. |

**Libros disponibles (en/):** Anatomy_Gray · Biochemistry_Lippincott · Cell_Biology_Alberts · First_Aid_Step1 · First_Aid_Step2 · Gynecology_Novak · Histology_Ross · Immunology_Janeway · InternalMed_Harrison · Neurology_Adams · Obstentrics_Williams · Pathology_Robbins · Pathoma_Husain · Pediatrics_Nelson · Pharmacology_Katzung · Physiology_Levy · Psichiatry_DSM-5 · Surgery_Schwartz

### 10 muestras representativas — Questions

| # | Origen | question (resumen) | answer (resumen) | meta_info | answer_idx |
|---|--------|-------------------|-----------------|-----------|------------|
| 1 | US | A 23-year-old pregnant woman at 22 weeks with burning urination — best treatment? | Nitrofurantoin | step2&3 | E |
| 2 | US | A 3-month-old baby died suddenly at night — which precaution could have prevented it? | Placing infant supine on firm mattress | step2&3 | A |
| 3 | US | A mother brings a 3-week-old infant with bilious vomiting — embryologic error? | Abnormal migration of ventral pancreatic bud | step1 | A |
| 4 | US | A 20-year-old woman presents with menorrhagia for the past… | *(continuación en archivo)* | step2&3 | — |
| 5 | Taiwan | 下列何者不是病人長期臥床不動（immobilization）之後的生理機能反應？ | 韌帶鬆弛，延展性增加 | taiwanese_test_Q | C |
| 6 | Taiwan | 人體內，vitamin D3轉為25-hydroxycholecalciferol主要是在下列何種器官中進行？ | liver | taiwanese_test_Q | C |
| 7 | Mainland | 坐位腰椎穿刺，脑脊液压力正常值是（　　）。 | 80～180 mmH₂O (0.78～1.76 kPa) | 第三部分 精神神经系统疾病 | B |
| 8 | Mainland | 男性儿童，左肱骨骨折急诊就医。小夹板，前臂高度肿胀，手部青凉，麻木无力，若不及时处理，其最可能的后果是？ | 缺血性肌挛缩 | 第四部分 迅骨系统疾病 | E |
| 9 | Mainland | 对疫疾病人做生物学检查，下列哪项是错误的？ | 取黏液性或血性粪便涂片，做闰标本接种于肠道选择培养基培 | 第三部分 *(sección)* | — |
| 10 | US *(textbook)* | *Anatomy_Gray.txt*: "What is anatomy? Anatomy includes those structures that can be seen grossly…" | *(texto libre, sin respuesta)* | — | — |

### Ejemplo completo — US (registro #1)
```json
{
  "question": "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination...",
  "answer": "Nitrofurantoin",
  "options": {
    "A": "Ampicillin",
    "B": "Ceftriaxone",
    "C": "Ciprofloxacin",
    "D": "Doxycycline",
    "E": "Nitrofurantoin"
  },
  "meta_info": "step2&3",
  "answer_idx": "E"
}
```

### Ejemplo completo — Mainland (registro #7)
```json
{
  "question": "坐位腰椎穿刺，脑脊液压力正常值是（　　）。",
  "options": {
    "A": "190～220 mmH₂O (1.86～2.16 kPa)",
    "B": "80～180 mmH₂O (0.78～1.76 kPa)",
    "C": "50～70 mmH₂O (0.49～0.69 kPa)",
    "D": "230～250 mmH₂O (2.25～2.45 kPa)",
    "E": "260～280 mmH₂O (2.55～2.74 kPa)"
  },
  "answer": "80～180 mmH₂O (0.78～1.76 kPa)",
  "meta_info": "第三部分　精神神经系统疾病",
  "answer_idx": "B"
}
```

---

## 3. Dataset: `multiclinsum_large-scale_train_es/`

**Descripción:** Dataset a gran escala de **resumenes de casos clínicos en español**. Cada caso contiene el texto clínico completo y su resumen correspondiente, listos para tareas de sumarización automática.

**Formato:** Texto plano (`.txt`), un archivo por caso clínico  
**Idioma:** Español  
**Tamaño:** 25 902 casos clínicos

### Estructura de directorios

```
multiclinsum_large-scale_train_es/
└── multiclinsum_large-scale_train_es/
    ├── fulltext/
    │   └── multiclinsum_ls_es_{n}.txt        ← Texto clínico completo
    └── summaries/
        └── multiclinsum_ls_es_{n}_sum.txt    ← Resumen del caso
```

### Esquema

| Elemento | Descripción |
|----------|-------------|
| **fulltext** | Texto narrativo sin estructura JSON. Describe datos demográficos del paciente, motivo de consulta, exploración física, pruebas diagnósticas (TC, MRI, laboratorio), tratamiento y evolución. |
| **summaries** | Párrafo(s) corto(s) con el resumen clínico del caso. |
| **Nombrado** | El par fulltext/summary comparte el mismo `n` numérico (`{n}.txt` ↔ `{n}_sum.txt`) |

### 10 muestras representativas

| # | Archivo | Contenido (resumen del caso) |
|---|---------|------------------------------|
| 1 | `multiclinsum_ls_es_1.txt` | Hombre japonés, 53 años, paraplejia incompleta súbita al levantar peso. MRI: masa epidural posterior en T9-T10. Cirugía de urgencia: laminectomía + fusión posterior. Diagnóstico final: hernia de disco torácica migrada. |
| 2 | `multiclinsum_ls_es_10.txt` | Hombre caucásico, 80 años, dolor abdominal agudo. TC: necrosis pancreática > 30 %. EUS-FNA: Citrobacter freundii. Tratamiento: ciprofloxacina oral 6 semanas. Resolución completa a 6 meses. |
| 3 | `multiclinsum_ls_es_100.txt` | Hombre iraní, 32 años, ataque tónico-clónico generalizado. Antecedentes de ascitis crónica. TC/MRI: focos multifocales + lesión prostática. Biopsia: granulomas necrotizantes → tuberculosis miliar. Tratamiento: HRZE 12 meses. |
| 4 | `multiclinsum_ls_es_1000.txt` | *(pendiente de muestra)* |
| 5 | `multiclinsum_ls_es_10000.txt` | *(pendiente de muestra)* |
| 6 | `multiclinsum_ls_es_1001.txt` | *(pendiente de muestra)* |
| 7 | `multiclinsum_ls_es_1002.txt` | *(pendiente de muestra)* |
| 8 | `multiclinsum_ls_es_1003.txt` | *(pendiente de muestra)* |
| 9 | `multiclinsum_ls_es_1004.txt` | *(pendiente de muestra)* |
| 10 | `multiclinsum_ls_es_1005.txt` | *(pendiente de muestra)* |

> Las muestras marcadas como *(pendiente)* se rellenan ejecutando la celda de código siguiente.

### Ejemplo completo — Summary (`multiclinsum_ls_es_10000_sum.txt`)
```
Presentamos a un paciente con dolor torácico, biomarcadores positivos de necrosis
miocárdica y bloqueo de rama derecha nuevo aislado en el ECG. Se le diagnosticó IAM,
pero no se le practicó una terapia de reperfusión urgente al no haber elevaciones del
segmento ST o un nuevo bloqueo de rama izquierda. Sin embargo, la angiografía demostró,
en última instancia, una oclusión coronaria completa.
```

### Ejemplo completo — Fulltext (`multiclinsum_ls_es_1.txt`, fragmento)
```
Un hombre japonés de 53 años de edad y con buen estado de salud, experimentó un inicio
repentino de paraplejia incompleta después de levantar un objeto pesado. El examen físico
reveló que su fuerza motora era de grado 0/5 en la extremidad inferior derecha y 0-2/5
en la extremidad inferior izquierda. Se observó pérdida sensorial por debajo del nivel
del ombligo. También se presentó incontinencia urinaria. La resonancia magnética (MRI)
reveló una masa epidural posterior que comprimía la médula espinal a nivel T9-T10...
```

---

## Resumen comparativo de datasets

| Dataset | Tarea NLP | Formato | Idioma(s) | Tamaño aprox. |
|---------|-----------|---------|-----------|---------------|
| `data/` | Medical QA (MCQ con explicaciones) | JSONL | Inglés | ~192 k preguntas |
| `data_clean/` | Medical QA multilingüe + RAG sobre textbooks | JSONL + TXT | EN / ZH-Hans / ZH-Hant | ~60 k preguntas + 18 libros |
| `multiclinsum_large-scale_train_es/` | Sumarización de casos clínicos | TXT pairs | Español | 25 902 pares |


---

### 10 muestras aleatorias — multiclinsum_large-scale_train_es

| # | Archivo (fulltext) | Fragmento fulltext | Resumen |
|---|---|---|---|
| 1 | `multiclinsum_ls_es_5543.txt` | Una mujer de 67 años de edad, naturalmente sana, fue admitida en el departamento de medicina general por un historial de dos semanas de fiebre, malestar, dolor de garganta y enzimas hepato-biliares elevadas. Se inició un… | Una mujer de 67 años presentó fiebre, malestar general y lesión renal aguda con proteinuria y hematuria que precisó hemodiálisis. Se le diagnosticó enfermedad por anticuerpos anti-GBM, basándose en el… |
| 2 | `multiclinsum_ls_es_13281.txt` | Una mujer de 67 años experimentó náuseas y vómitos no expulsivos acompañados de molestias en la parte superior del abdomen, distensión abdominal, fatiga y falta de apetito hace aproximadamente 20 días. El vómito era cont… | Se diagnosticó a una mujer de 67 años un adenocarcinoma gástrico intramucoso tras una biopsia en el hospital local hace tres semanas y luego visitó nuestro hospital para recibir tratamiento adicional.… |
| 3 | `multiclinsum_ls_es_10735.txt` | Se diagnosticó cáncer pulmonar de células escamosas, en estadio cT1miN0M0 y estadio IA1, a un hombre de 73 años de edad con antecedentes de hipertensión, diabetes, neumoconiosis y enfermedad pulmonar obstructiva crónica.… | Se diagnosticó a un hombre de 73 años con cáncer de pulmón de células escamosas (estadio IA1 cT1miN0M0). Debido a la neumoconiosis, se observó una extensa infiltración de los ganglios linfáticos en la… |
| 4 | `multiclinsum_ls_es_8556.txt` | Un hombre de 57 años se presentó en el departamento de urgencias con quejas de aumento de la fatiga y falta de aire durante el mes anterior. En la presentación inicial, informó de empeoramiento de la disnea junto con pre… | Un hombre de 57 años con antecedentes de trasplante de corazón 6 años antes, presentó un mes de fatiga severa y falta de aire. Sus valores de laboratorio al ingreso fueron notables por una pancitopeni… |
| 5 | `multiclinsum_ls_es_18109.txt` | Una mujer de 44 años con antecedentes de 27 años de enfermedad de Behcet se presentó en el hospital con dificultad respiratoria secundaria a un colapso recurrente de su lóbulo inferior izquierdo. Como consecuencia de su … | Presentamos el caso de una mujer de 44 años con enfermedad de Behcet, úlceras bucales y genitales con síndrome de cartílago inflamado (MAGIC) y un tronco de elefante congelado aórtico (FET) que se pre… |
| 6 | `multiclinsum_ls_es_1722.txt` | Se diagnosticó a un paciente de 37 años con meningioma de células claras atípico recurrente del agujero occipital, columna cervical (nivel C1-C2) y columna lumbar, confirmado por resonancia magnética (MRI). La MRI preope… | Este estudio reporta un caso raro de migración postoperatoria de un VPS, en el que el catéter distal sale de la cavidad abdominal a través de una hernia de Grynfeltt. Esta condición no fue descubierta… |
| 7 | `multiclinsum_ls_es_16581.txt` | Se evaluó a un niño de 8 meses de edad por falta de crecimiento y retraso en el desarrollo. Nació a las 38 semanas de gestación (peso al nacer de 4090 g, por encima del percentil 90).… | Informamos sobre un niño de 8 meses de edad que fue evaluado por falta de crecimiento, estreñimiento y retraso en el desarrollo. Los síntomas comenzaron después de la introducción de gluten en la diet… |
| 8 | `multiclinsum_ls_es_14112.txt` | Una mujer Han china de 68 años con un historial médico claro fue admitida en nuestro hospital porque había reportado heces sanguinolentas intermitentes durante aproximadamente 6 meses que habían empeorado durante casi 11… | Reportamos el caso de una mujer de 68 años con neoplasias rectales y SIT diagnosticados mediante biopsia de colonoscopia electrónica y tomografía computarizada (TC) mejorada, que mostró que había una … |
| 9 | `multiclinsum_ls_es_8405.txt` | Una mujer de 30 años se sometió a una oforectomía izquierda, después de lo cual se le diagnosticó un carcinoma atípico del ovario. Dos años después de este diagnóstico, se sometió a una cesárea y se encontró un carcinoma… | Presentamos un caso de carcinoma ovárico atípico relacionado con MEN1 que se manifestó como la primera manifestación de la enfermedad en una mujer de 30 años. Después de dos años, se diagnosticó incid… |
| 10 | `multiclinsum_ls_es_1302.txt` | Un hombre de 53 años de edad, con antecedentes de 10 años de episodios repetidos de pérdida transitoria de conciencia tras girar la cabeza hacia la derecha. No describió otros síntomas y no tenía antecedentes de traumati… | Un hombre de 53 años de edad, con un historial de 10 años de episodios repetidos de pérdida transitoria de conciencia después de girar el cuello hacia la derecha. Aunque la resonancia magnética no mos… |
