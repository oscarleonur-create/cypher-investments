# Plan: amplitud — encontrar nombres nuevos que merezcan la profundidad

Aceptado el 2026-09-27 ("dale arranca con B0"). Las decisiones 1–4 del final
siguen abiertas; B0 no depende de ninguna.

## La idea

Más nombres no es más edge: es más ruido. El 4,7% de los días-símbolo cruza
|z|≥2 (`swing-entry-plan.md`); sobre 2.000 nombres serían ~2.000
"oportunidades" al mes. Los datos sobran. Lo escaso es **tu atención** y la
**profundidad** (20 consultas Tavily al día, lecturas del modelo). Por eso el
diseño es un embudo con presupuesto, y la aguja se define antes de buscarla:
un patrón con tasa base medible, no "algo interesante".

Lo que ya existe no cubre esto: el escáner intradía busca movers en todo el
mercado para operar hoy; la profundidad (zona, lectura, distress, noticias)
solo corre sobre nombres conocidos. Falta encontrar **nombres nuevos** para
un horizonte de semanas a meses.

## Medido el 2026-09-27

| fuente | resultado |
|---|---|
| Directorio Nasdaq Trader | 7.515 emisiones listadas no-ETF, 0,8 s |
| SEC XBRL `frames` | ingresos CY2026Q2 de ~4.000 empresas en 2 llamadas, <1 s |
| yfinance masivo | 1.500 nombres × 1 año en 103 s |
| yfinance, segunda descarga minutos después | **0 de 1.500 — bloqueado** |
| `frames`, 48 llamadas | 5.765 empresas con ingresos, 4.101 con TTM; los años fiscales no calendario pierden el Q4 fiscal (MSFT, COST) |

## El embudo

| etapa | qué | tamaño |
|---|---|---|
| **E0 Universo** | directorio → acciones comunes de emisores SEC → precio ≥$5, ADV mediano ≥$10M, ≥250 sesiones | ~7.500 → ver B0 |
| **E1 Señales baratas** | por familias independientes, sobre datos masivos | cientos/día |
| **E2 Convergencia** | candidato solo si ≥2 familias en 30 días; una sola se registra, no se muestra | decenas |
| **E3 Profundidad** | lo existente, con presupuesto | ~5/día |
| **E4 Escritorio** | tú: subir a watchlist/Swing, descartar con motivo, posponer | ≤5/semana |

Familias E1 (evidencia de deriva a 1–6 meses — lo contrario de la
sobrerreacción, que tus datos refutaron):

| familia | señal | fuente |
|---|---|---|
| F fundamentales | aceleración de ingresos, inflexión de margen | SEC `frames` |
| P precio | fuerza relativa 6/12m, máximo de 52 semanas, ruptura con volumen vs su σ | barras diarias |
| I informados | compras de insiders en grupo (Form 4 código P), 13D | índice diario EDGAR |
| E eventos | 8-K 2.02 con reacción ≥2σ sostenida, 1.01 | índice diario EDGAR + clasificador |
| T adyacencia | proveedores/clientes nombrados en los 10-K de tus posiciones; co-movimiento residual | EDGAR + sensibilidades |

Las anomalías publicadas pierden 30–50% tras publicarse (McLean–Pontiff):
ninguna se da por buena hasta medirla aquí.

Exclusiones duras reutilizadas: going concern, dilución, halts, reverse
split reciente, resultados en <5 sesiones.

## Medición antes que escritorio

Cada impacto E1, candidato E2 y decisión E4 se guarda con sello de reglas y
retornos a 20/60/120 sesiones, contra un nulo de nombres al azar del mismo
E0, mismo día, emparejados por sector y tamaño. UNDETERMINED por defecto.
E1/E2 son replayables sobre 2–4 años antes de mostrarte nada. Tus descartes
también se miden. Sesgo de supervivencia: el directorio es el de hoy.

## Lo que se niega a hacer

Valor razonable; "comprar"; añadir a la watchlist solo; interrumpir (va al
digest); mostrar impactos de una sola familia; rankear por retorno esperado.

## Fases

- **B0** — E0, almacén de barras diarias (archivo propio), almacén de
  ingresos/acciones por `frames` con relleno por `companyconcept`.
- **B1** — familias F y P + replay de 2 años contra el nulo. Decide si lo
  demás existe, y mide cuántos candidatos/día salen de verdad.
- **B2** — familias I y E (índice diario EDGAR para todos los emisores).
- **B3** — familia T.
- **B4** — escritorio en la UI.

## Decisiones abiertas

1. Horizonte de la aguja. Recomendado: 1–6 meses.
2. Universo agnóstico o solo AI-buildout. Recomendado: agnóstico, con la
   adyacencia como una familia.
3. Presupuesto de atención. Propuesto: ≤5/semana.
4. B1 (replay + nulo) antes de mostrar nada. Recomendado: sí.
