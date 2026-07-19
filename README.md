# IBM AI Enterprise Workflow — Specialization Projects

![Coursera](https://img.shields.io/badge/Coursera-IBM-0056D2)
![Focus](https://img.shields.io/badge/Focus-AI%20Workflow-informational)
![Language](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Tools](https://img.shields.io/badge/pandas%20%C2%B7%20scikit--learn%20%C2%B7%20statsmodels-orange)

Coursework, notebooks, and deliverables for IBM's six-course
[**AI Enterprise Workflow** specialization](https://www.coursera.org/specializations/ibm-ai-workflow) —
a practitioner's path through the full data-science lifecycle inside a real business context,
from framing the problem and ingesting data, to modelling, deployment, and production monitoring.

---

## The through-line: the AAVAIL case study

Every course builds on the same fictional client — **AAVAIL**, a media/streaming subscription
company operating in the US and Singapore. Rather than isolated toy exercises, the assignments
stack into one continuous engagement: understand AAVAIL's data, find a business opportunity,
investigate it rigorously, and hand stakeholders a decision. This repo mirrors that arc.

**Headline result from the case work:** AAVAIL's Singapore market churns at **~60%** vs **~15%**
in the US. The analysis traces this to a **price-value mismatch in the premium plan tiers** —
not to low engagement — confirmed by two independent models and packaged into a stakeholder
storyboard.

---

## Courses

| # | Course | Theme |
|---|--------|-------|
| 1 | Business Priorities and Data Ingestion | Scoping the business problem, building a reusable data-ingestion pipeline |
| 2 | Data Analysis and Hypothesis Testing | EDA, distributions, statistical testing (t-tests, chi-square, effect sizes) |
| 3 | Feature Engineering and Bias Detection | Transformations, missing-data imputation, fairness/bias checks |
| 4 | Machine Learning, Visual Recognition and NLP | Model selection and evaluation across ML, CV, and NLP tasks |
| 5 | Enterprise Model Deployment | Packaging models as services (APIs, containers, workflow orchestration) |
| 6 | AI in Production | Monitoring, logging, performance/drift management, feedback loops |

> Course titles follow IBM's published curriculum; verify against the current Coursera page,
> as the specialization is periodically updated.

---

## Repository structure

```
.
├── README.md
├── data/
│   ├── aavail-data-visualization.csv        # raw AAVAIL customer data
│   └── aavail-imputed.csv                    # cleaned dataset (post-imputation)
├── notebooks/
│   ├── aavail-full-analysis.ipynb            # end-to-end: ingest -> impute -> test -> model
│   └── aavail-full-analysis-slides.ipynb     # RISE-tagged presentation version
├── src/
│   └── aavail_visualizations.py              # reusable plotting blocks
├── deliverables/
│   ├── aavail-singapore-churn-storyboard.html  # stakeholder storyboard
│   ├── aavail-market-visualization.png         # 4-panel market snapshot
│   └── aavail-full-analysis.pdf                # exported report
└── requirements.txt
```

---

## Key deliverables

- **Missing-data investigation & imputation** — diagnosed the missingness *mechanism*
  (MCAR vs MAR) and applied mechanism-specific fills (proportional within-country for the
  categorical column, group median for the numeric one), validated to shift distributions by
  < 0.3 pp.
- **Hypothesis testing** — chi-square with Cramer's V for associations; Mann-Whitney U with
  Cohen's d for group comparisons, reporting effect sizes alongside p-values.
- **Churn driver analysis** — logistic regression (interpretable odds ratios) cross-checked
  with a random forest (5-fold ROC-AUC ~ 0.74, permutation importance).
- **Segmentation** — K-means to locate the concentrated at-risk customer group.
- **Communication** — a self-contained HTML storyboard and an exported PDF report aimed at
  non-technical stakeholders.

---

## Skills & tools

`Python` · `pandas` · `NumPy` · `SciPy` · `statsmodels` · `scikit-learn` · `Matplotlib` ·
`seaborn` · `Jupyter` · `RISE`

Data ingestion pipelines · exploratory data analysis · statistical hypothesis testing ·
missing-data imputation · bias/fairness awareness · supervised modelling · unsupervised
segmentation · model evaluation · data storytelling.

---

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

# run the analysis
jupyter notebook notebooks/aavail-full-analysis.ipynb

# open the stakeholder storyboard
open deliverables/aavail-singapore-churn-storyboard.html
```

**`requirements.txt`**
```
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
seaborn
jupyter
RISE
```

---

## Author

**Menna Allah Samy** — Data Scientist & ML Engineer
Completed as part of the IBM AI Enterprise Workflow specialization on Coursera.

## Acknowledgements

Course materials and the AAVAIL case study © IBM, delivered via Coursera. This repository
contains original solution work; it is intended as a learning portfolio and not a substitute
for completing the assignments.
