# Manual de Uso Básico

**Credit Risk Predictor** — Sistema de Predicción de Riesgo Crediticio con Remuestreo Híbrido

> Dirigido a: Científico de Datos y Analista de Riesgo Financiero
> Versión 1.0 · USIL 2026

---

## 1. Introducción

Este manual explica cómo usar la aplicación **Credit Risk Predictor** sin necesidad de conocimientos de programación.

> Si su objetivo es instalar el sistema o modificar el código, consulte el [`README.md`](./README.md) en su lugar.

### 1.1 Qué hace esta aplicación

Permite cargar un archivo de datos crediticios (CSV), analizarlo, balancear las clases de impago/buen crédito, entrenar un modelo predictivo y entender por qué el modelo toma cada decisión.

### 1.2 Cómo abrir la aplicación

1. Abra una terminal en la carpeta del proyecto.
2. Ejecute el comando:
   ```bash
   streamlit run app.py
   ```
3. Se abrirá automáticamente en su navegador en la dirección `http://localhost:8501`

---

## 2. Recorrido paso a paso

La aplicación se usa siguiendo 6 pantallas en orden, visibles en el menú lateral izquierdo. No puede saltar a una pantalla posterior sin completar la anterior.

### Paso 1 — Cargar datos (RF1)

- Haga clic en el área de carga y seleccione su archivo CSV.
- El sistema le pedirá elegir cuál columna es la variable objetivo (la que indica impago o buen crédito).
- Si su columna tiene texto en lugar de 0 y 1 (por ejemplo `good` y `bad`), no se preocupe: el sistema la convierte automáticamente.
- Presione **Confirmar carga y continuar** cuando esté conforme.

> **Aviso:** Si ve una advertencia de columnas con muchos valores nulos, puede continuar sin problema: el sistema completa esos valores automáticamente con la mediana o el valor más común.

### Paso 2 — Revisar el análisis exploratorio (RF2)

- Vea cuántos clientes de cada tipo (buen crédito / impago) tiene su archivo.
- Revise el indicador de **Imbalance Ratio**: si es alto (mayor a 10), significa que sus datos están muy desbalanceados.
- Explore las pestañas de variables numéricas, correlaciones y calidad de datos para entender mejor su archivo.
- Presione **Preprocesar y continuar** para pasar al siguiente paso.

### Paso 3 — Elegir técnica de balanceo (RF3)

- Seleccione una técnica de remuestreo. Si no está seguro, deje la opción recomendada por defecto: **SMOTE-ENN + Tomek (híbrido)**.
- Presione **Previsualizar distribución** para ver cómo quedarán balanceadas sus clases antes de continuar.
- Confirme la configuración para avanzar al entrenamiento.

### Paso 4 — Entrenar el modelo (RF4)

- Elija un algoritmo: **Random Forest** es una buena opción por defecto para la mayoría de los casos.
- Presione **Entrenar modelo** y espere a que la barra de progreso llegue al 100%.
- El entrenamiento normalmente tarda menos de 2 minutos.

### Paso 5 — Ver resultados (RF5)

- Revise la tabla comparativa: muestra si el modelo mejoró después de balancear los datos.
- Los números en verde significan mejora; los números en rojo significan que esa métrica empeoró.
- Verifique que aparezcan las marcas de cumplimiento de los 3 indicadores clave (KPIs) del proyecto.

### Paso 6 — Entender las decisiones del modelo (RF6)

- Vea el gráfico de **importancia global**: muestra qué variables influyen más en las decisiones del modelo.
- Seleccione un cliente específico para ver por qué el modelo lo aprobó o lo rechazó.
- Use esta información para explicar una decisión a un cliente o a un auditor.

---

## 3. Preguntas frecuentes

| Pregunta | Respuesta |
|---|---|
| ¿Puedo usar cualquier archivo CSV? | Sí, siempre que tenga al menos una columna que indique impago o buen crédito (puede ser 0/1 o texto como `good`/`bad`). |
| ¿Se guardan mis datos después de cerrar la app? | No. El sistema no guarda información en ninguna base de datos. Todo se borra al cerrar la sesión. |
| ¿Qué hago si el sistema muestra un error? | Revise que su archivo CSV tenga una columna objetivo válida. Si el error persiste, contacte al equipo de desarrollo. |
| ¿Cuál técnica de remuestreo debería elegir? | El enfoque híbrido (SMOTE-ENN + Tomek) es la opción recomendada para la mayoría de los casos. |
| ¿Puedo volver a una pantalla anterior? | Sí, use el menú lateral. Si cambia una configuración anterior, los resultados posteriores se reiniciarán automáticamente. |

---

## 4. Soporte

Para reportar errores o solicitar ayuda, contacte al equipo de desarrollo a través del repositorio del proyecto en GitHub:

**Repositorio:** [https://github.com/lMiaul/pf_TISW/tree/develop](https://github.com/lMiaul/pf_TISW/tree/develop)

> Para detalles técnicos de instalación, consulte el archivo [`README.md`](./README.md) del repositorio.

---

*Universidad San Ignacio de Loyola · Tópicos de Ingeniería de Software · Lima, 2026*