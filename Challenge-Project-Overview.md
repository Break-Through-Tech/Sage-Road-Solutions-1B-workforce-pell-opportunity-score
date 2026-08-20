---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | The project utilizes standard scikit-learn and XGBoost libraries which are fully supported in Google Colab environment. |
| Data Readiness | 🟡 | While public, the datasets (IPEDS, BLS, O*NET, Census) involve complex relational merging across multiple schemas, which will be time-consuming for students. |
| Resource Check | 🟢 | Fits within standard CPU-based compute; no GPUs or paid APIs required. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
This project offers a strong opportunity to apply ensemble methods to high-impact social data. To succeed, first, simplify the data ingestion phase by providing a pre-merged 'base' file to avoid 10 weeks of cleaning. Second, implement a strict time-based validation holdout set to ensure the model generalizes to future credential cycles. I recommend we proceed with these scoping adjustments to ensure the dashboard remains a feasible deliverable.

---

# Workforce Pell Opportunity Score: Predicting High-Impact Short-Term Education Programs

**Company / Org:** Sage Road Solutions   
**Challenge Advisor:** Beth Davis, bethdavisnc@gmail.com  
**AI Studio Coach:** Hrushikesh Shetty, hrushikesh.shetty@breakthroughtech.org  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Sage Road Solutions
This project focuses on the intersection of higher education and labor market analytics, aiming to optimize the impact of Workforce Pell grants. The team will collaborate to build data-driven tools that assist policymakers and educational institutions in identifying short-term programs that lead to successful workforce integration.

---

## 🎯 The Challenge
### Project Summary
In this project, students will use publicly available education, workforce, and labor market data—including IPEDS, College Scorecard, BLS Occupational Employment Statistics, O*NET, and U.S. Census data—to develop an explainable machine learning model that estimates the likelihood that short-term education programs align with Workforce Pell priorities and regional workforce demand. Using feature engineering, gradient-boosted decision trees, clustering, and SHAP-based explainability, the team will identify the institutional, demographic, and labor market characteristics associated with strong workforce outcomes. The resulting Opportunity Score and interactive dashboard will help colleges, workforce agencies, employers, and policymakers identify where investments in short-term credential programs are most likely to improve employment outcomes and expand equitable access to high-quality training.

### Success Criteria

_Evaluation metrics:_ 
- Data source quality and selection criteria
- Model accuracy, precision, recall, and F1 score for classification tasks.
- ROC-AUC where appropriate.
- Cross-validation performance to assess generalizability.
- Model explainability 

_Business success metrics_   
- Ability to identify institutional and regional characteristics associated with Workforce Pell readiness.
- Generation of an interpretable Workforce Pell Opportunity Score for institutions or programs.
- Production of actionable recommendations that education leaders, workforce agencies, and policymakers could use to prioritize investments.
- Delivery of a reproducible open-source workflow and interactive dashboard suitable for future expansion.

A successful project will demonstrate that publicly available education and labor market data can be combined to produce meaningful, explainable predictions that support strategic planning for Workforce Pell implementation.

### Stretch Goals

Potential extensions include:

- Developing a recommendation engine that suggests new certificate programs based on regional workforce demand.
- Forecast future labor market demand using time-series analysis.
- Incorporate additional public datasets such as state workforce projections or broadband access.
- Build a conversational AI assistant using a Retrieval-Augmented Generation (RAG) architecture that answers questions such as: "Which Illinois community colleges have the greatest Workforce Pell opportunity?"; "Why did this institution receive a high opportunity score?"
- Deploy the dashboard as a lightweight web application.
- Evaluate model fairness across institution types (community colleges, Minority-Serving Institutions, HBCUs, HSIs, rural institutions) to identify potential sources of algorithmic bias.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Data Discovery, Integration, and Exploratory Analysis | • Identify and acquire public datasets (IPEDS, College Scorecard, BLS Occupational Employment Statistics, O*NET, U.S. Census ACS, and CIP code crosswalks).<br>• Design a unified data model linking institutions, academic programs, occupations, and regional labor markets.<br>• Perform exploratory data analysis to identify missing data, regional trends, and candidate predictive features.<br>• Develop an initial feature set and establish baseline models for comparison.<br>• **Deliverable:** Clean, integrated analytical dataset, feature dictionary, exploratory dashboard, and baseline predictive model. |
| October | Machine Learning Model Development | • Engineer features representing institutional quality, labor market demand, demographic characteristics, and regional economic conditions.<br>• Develop and compare multiple machine learning models (Random Forest, XGBoost, Gradient Boosting, Logistic Regression).<br>• Evaluate model performance using cross-validation.<br>• Apply explainability techniques to identify the most influential variables.<br>• **Deliverable:** Candidate prediction models with documented performance metrics and explainability analysis. |
| November | Visualization, Validation, and Decision Support Tool | • Select the best-performing model.<br>• Build an interactive dashboard allowing users to explore Workforce Pell Opportunity Scores by institution, region, or program.<br>• Validate results against known Workforce Pell criteria and expert review.<br>• Prepare technical documentation, reproducible notebooks, and final presentation materials.<br>• **Deliverable:** Final predictive model, interactive dashboard, GitHub repository, technical documentation, and executive presentation. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** Publicly available education and labor market data (IPEDS, College Scorecard, BLS, O*NET, Census ACS)  
**Format:** CSV / Structured Relational  
**Size:** under 1gb  
**Location:** 
- IPEDS: https://nces.ed.gov/ipeds/use-the-data   
- College Scorecard: https://collegescorecard.ed.gov/data/   
- Bureau of Labor Statistics Occupational Employment and Wage Statistics: https://www.bls.gov/oes/   
- O*NET Data: https://www.onetcenter.org/database.html   
- U.S. Census American Community Survey: https://www.census.gov/programs-surveys/acs/data.html
- Post Secondary Employment Outcomes [https://lehd.ces.census.gov/applications/pseo/?](https://lehd.ces.census.gov/data/pseo_documentation.html) 

### Key Details
- **What's in the data:** IPEDS is the master list of U.S. colleges - where each one is located, what type it is, and what degrees and certificates it awarded, broken down by subject area (CIP code). College Scorecard adds outcomes: how many students finished, what they earned afterward, and how much debt they took on. BLS OEWS tells you, for each job type (SOC code) in each region of the country, how many people work in it and what it pays. O*NET describes the jobs themselves - what training they need, what skills they use. Census ACS fills in the local picture: income, education levels, and employment for each county.

- **Known limitations and preprocessing:** The biggest one is that **there's no answer key**. No dataset tells you which programs actually qualify for Workforce Pell, and two of the things states care most about - whether graduates got jobs, and whether the program was worth what it cost - aren't in any of these files. So the team will need to decide early what they're actually predicting or scoring. Coverage is also limited: Workforce Pell covers short programs (8 to 15 weeks), and most training that short is non-credit, which IPEDS doesn't track - that may mean scoring colleges rather than individual programs. The trickiest technical piece is connecting education data to job data: college programs use CIP codes and jobs use SOC codes, and the crosswalk between them isn't one-to-one, so the team needs a clear rule for handling programs that map to several jobs. Location needs a rule too, since IPEDS gives one campus address, OEWS uses metro areas, and ACS uses counties. Finally, expect missing values - both Scorecard and OEWS hide numbers when the group is too small, which happens most often at rural colleges.
 
---

## 🛠️ Suggested Approach

**ML Problem Type:** Classification, Regression, Clustering, Recommendation Systems, Large Language Models (LLMs)/ Generative AI  

**Recommended Libraries:**
- pandas
- numpy
- scikit-learn
- XGBoost / LightGBM
- SHAP
- matplotlib / seaborn
- plotly
- streamlit

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1
- ROC-AUC
- RMSE, MAE, R²
- Silhouette score (for clustering)
- Cross-validation performance

  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Providing Public Workforce Services to Job Seekers: 30-Month Impact Findings on the WIA Adult and Dislocated Worker Programs Final Report **
https://www.dol.gov/resource-library/providing-public-workforce-services-job-seekers-30-month-impact-findings-wia-adult 

**Background Reading:**
Bipartisan Workforce Pell Act fact sheet 
https://democrats-edworkforce.house.gov/imo/media/doc/bipartisan_workforce_pell_act_fact_sheet.pdf Helps explain where the policy architecture came from: short programs, employer/labor-market alignment and explicit quality safeguards were central to the proposals well before enactment.

Variability in approach 
(https://workforcepell.substack.com/p/workforce-pell-an-early-look-at-states?utm_source=substack&publication_id=6494247&post_id=207040077&utm_medium=email&utm_content=share&utm_campaign=email-share&isFreemail=true&r=6w2gc&triedRedirect=true)

U.S. Department of Education — Final Workforce Pell Rule / announcement
https://www.ed.gov/about/news/press-release/us-department-of-education-issues-final-rule-create-new-workforce-pell-grant-program
This is now the authoritative starting point. Workforce Pell took effect July 1, 2026, and covers qualifying short-term programs beginning at 8 weeks. The eligibility structure is deliberately performance-based rather than merely based on program length.
ED final-rule announcement

ED Workforce Pell Final Rule Fact Sheet 
https://www.ed.gov/media/document/workforce-pell-grant-final-rule-fact-sheet-114075.pdf
Among other requirements, eligible programs generally must fall in the 150–599 clock-hour / 8-to-less-than-15-week range.
Workforce Pell final-rule fact sheet

Federal Student Aid implementation material
https://fsapartners.ed.gov/sites/default/files/attachments/2026-03/Pell%20Eligibility%20for%20Workforce%20Programs.pdf
 Institutions were permitted to begin applying for ED approval of eligible workforce programs on July 1, 2026.
FSA: Pell eligibility for Workforce Programs

Why this is hard. Long report from the Urban Institute 
https://www.urban.org/research/publication/should-federal-government-fund-short-term-postsecondary-certificate-programs

A perspective that the program is bad: 
https://www.newamerica.org/insights/the-bad-policy-that-wont-die-and-has-gotten-worse-short-term-pell/

Postsecondary Commission Texas Study - longitudinal and wage data 
https://www.texas-psc.org/bachelors/programmatic-cohorts 

https://www.urban.org/research/publication/how-many-short-term-training-programs-would-gain-access-pell-grants-under-new-proposal


**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
Data from Urban Institute analysis  (https://github.com/UrbanInstitute/The-Learning-Curve/tree/main/2023_03_29-Short_term)
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* email: bethdavisnc@gmail.com. I'm in Central Time, and emails are best before or after the work day. Please copy your teammates and AI Studio Coach on all emails. 
* N/A 
* I will aim to respond within 48 hours. If you have a specific time by which you would like me to respond, please include that information in the email. 

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
